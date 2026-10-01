import importlib.metadata
import os
import platform
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from seizure_v2.common import write_json, read_json, sha256, object_hash
from seizure_v2.data.prepare import audit_dataset, load_rows
from seizure_v2.data.normalization import fit_scaler
from seizure_v2.models import make_model
from seizure_v2.training.dataset import WindowDataset
from seizure_v2.evaluation.metrics import grouped_metrics, macro_average, select_threshold
from seizure_v2.recovery import checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, loader, device):
    model.eval()
    output = []
    with torch.inference_mode():
        for x, _ in loader:
            output.extend(torch.sigmoid(model(x.to(device))).cpu().numpy().tolist())
    return np.asarray(output)


def atomic_checkpoint(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train(root, config, output, resume=False, development=False, epochs=None, device="auto", num_workers=None):
    cfg = yaml.safe_load(Path(config).read_text())
    if epochs is not None:
        if not development:
            raise ValueError("Epoch override is only available for explicitly marked development runs")
        cfg["max_epochs"] = epochs
    if num_workers is not None:
        cfg["num_workers"] = num_workers
    report = audit_dataset(root, require_full=not development)
    output = Path(output)
    if (output / "frozen.json").exists() and not resume:
        raise ValueError("Run is frozen; use a new run directory")
    if (output / "last.pt").exists() and not resume:
        raise ValueError("Existing run; explicitly resume or use a new directory")
    state, saved_run = None, None
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if state["dataset_hash"] != report["dataset_hash"] or state["config"] != cfg:
            raise ValueError("Resume requires identical dataset and configuration")
        saved_run = read_json(output / "run.json")
        if saved_run["development"] != development:
            raise ValueError("Cannot change a run's development status")
        if (output / "frozen.json").exists():
            frozen = read_json(output / "frozen.json")
            if sha256(output / "best.pt") != frozen["checkpoint_sha256"] or sha256(output / "scaler.json") != frozen["scaler_sha256"]:
                raise ValueError("Frozen artifacts changed")
            return frozen
    output.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = load_rows(root, "train"), load_rows(root, "validation")
    if not train_rows or not val_rows:
        raise ValueError("Both patient-separated training and validation data are required")
    positives = sum(row["label"] for row in train_rows)
    if not 0 < positives < len(train_rows):
        raise ValueError("Training requires both classes")
    if {row["individual_id"] for row in train_rows} & {row["individual_id"] for row in val_rows}:
        raise ValueError("Patient leakage")
    device = choose_device(device)
    seed_everything(cfg["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    generator = torch.Generator().manual_seed(cfg["seed"])
    scaler = fit_scaler(root, train_rows, report["dataset_hash"])
    write_json(output / "scaler.json", scaler)
    model = make_model(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    weight = (len(train_rows) - positives) / positives
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    workers = cfg["num_workers"]
    loader_options = dict(batch_size=cfg["batch_size"], num_workers=workers, worker_init_fn=seed_worker,
                          pin_memory=device.type == "cuda")
    train_loader = DataLoader(WindowDataset(root, train_rows, scaler, random_access=True), shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(WindowDataset(root, val_rows, scaler), shuffle=False, **loader_options)
    start, best, stale, history = 0, -1., 0, []
    if state is not None:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        for value in optimizer.state.values():
            for key, item in value.items():
                if torch.is_tensor(item):
                    value[key] = item.to(device)
        start, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "torch", "scikit-learn", "pyedflib"]}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    run = {"config": cfg, "dataset_hash": report["dataset_hash"], "split_sha256": report["split_sha256"],
           "development": development, "device": str(device), "versions": versions,
           "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                        "cpu_count": os.cpu_count(), "cuda_version": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
           "git_commit": git.stdout.strip() or "uncommitted", "pos_weight": weight,
           "parameter_count": sum(p.numel() for p in model.parameters()),
           "determinism": "seeded; deterministic algorithms requested with warnings; cross-device bitwise identity is not guaranteed"}
    provenance_fields = ["git_commit", "versions", "hardware", "device"]
    previous_segments = [] if saved_run is None else saved_run.get("execution_segments", [
        {**{key: saved_run[key] for key in provenance_fields}, "first_epoch": 1}])
    run["execution_segments"] = [*previous_segments,
        {**{key: run[key] for key in provenance_fields}, "first_epoch": start + 1,
         "shuffled_cache_io": "MADV_RANDOM when supported; signal tensors and order unchanged"}]
    write_json(output / "run.json", run)
    for epoch in range(start, cfg["max_epochs"]):
        if stale >= cfg["patience"]:
            break
        began = time.monotonic()
        model.train()
        total_loss = 0.
        print(f"{cfg['model']} epoch {epoch + 1}: training {len(train_rows)} windows", flush=True)
        for batch_index, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            if batch_index % 200 == 0 or batch_index == len(train_loader):
                print(f"{cfg['model']} epoch {epoch + 1}: batch {batch_index}/{len(train_loader)} "
                      f"({time.monotonic() - began:.0f}s)", flush=True)
        print(f"{cfg['model']} epoch {epoch + 1}: validating {len(val_rows)} windows", flush=True)
        scores = predict(model, val_loader, device)
        value = macro_average(grouped_metrics(val_rows, scores, .5))["auprc_ap"]
        if value is None:
            raise ValueError("Validation patient AP is undefined")
        improved = value > best
        best, stale = (value, 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_rows),
                        "validation_patient_macro_ap": value, "seconds": time.monotonic() - began})
        state = {"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best": best, "stale": stale, "history": history,
                 "config": cfg, "dataset_hash": report["dataset_hash"],
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if improved:
            atomic_checkpoint(output / "best.pt", {"model_state": model.state_dict(), "config": cfg,
                                                    "dataset_hash": report["dataset_hash"], "epoch": epoch})
        atomic_checkpoint(output / "last.pt", state)
        write_json(output / "history.json", history)
        print(history[-1], flush=True)
        checkpoint(f"{cfg['model']}-epoch-{epoch + 1:02d}")
    best_state = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_state["model_state"])
    scores = predict(model, val_loader, device)
    selected = select_threshold(val_rows, scores, cfg["threshold_step"])
    frozen = {**run, **selected, "validation_patient_macro_ap": best,
              "checkpoint_sha256": sha256(output / "best.pt"), "scaler_sha256": sha256(output / "scaler.json"),
              "checkpoint_epoch": best_state["epoch"] + 1}
    write_json(output / "frozen.json", frozen)
    checkpoint(f"{cfg['model']}-frozen")
    return frozen


def freeze_study(runs, output, development=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Study is already frozen; do not overwrite after test evaluation")
    models = {}
    for path in map(Path, runs):
        frozen = read_json(path / "frozen.json")
        if frozen["development"] != development:
            raise ValueError("Cannot mix development and final runs")
        if sha256(path / "best.pt") != frozen["checkpoint_sha256"] or sha256(path / "scaler.json") != frozen["scaler_sha256"]:
            raise ValueError("Frozen artifacts changed")
        name = frozen["config"]["model"]
        if name in models:
            raise ValueError("Duplicate architecture")
        models[name] = {**frozen, "run_dir": os.path.relpath(path.resolve(), output.parent.resolve())}
    if set(models) != {"baseline", "residual"} or len({m["dataset_hash"] for m in models.values()}) != 1:
        raise ValueError("Both architectures must use exactly the same dataset")
    default = max(sorted(models), key=lambda name: models[name]["validation_patient_macro_ap"])
    study = {"development": development, "models": models, "default_model": default,
             "selection_partition": "validation", "selection_metric": "patient_macro_ap"}
    study["sha256"] = object_hash(study)
    write_json(output, study)
    return study
