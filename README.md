# Pediatric EEG · Research Explorer

**How well can a compact seizure detector generalize to a patient it has never encountered?**

V2 studies that question using original CHB-MIT EDF recordings, a frozen patient-independent split, a V1-style CNN, and one compact residual CNN. A browser demo makes both successful predictions and failure examples inspectable.

**Status:** the frozen full-cohort experiment and [public inference demo](https://pediatric-eeg-seizure-v2.onrender.com/) are complete and verified. Both models were trained and frozen before held-out evaluation. After the first evaluation artifacts were lost with an expired Colab runtime, the reports and demo were **regenerated on 2026-10-02 from the original frozen models and protocol**. No weights, scaler, threshold, split, preprocessing, architecture, hyperparameters, or model-selection decision changed. Every source/cache/manifest equivalence check passed before evaluation. Final artifacts were downloaded and hash-verified locally before website integration. The compact verified deployment bundle is published as [GitHub Release v2.0.0](https://github.com/iampenuel/pediatric-eeg-seizure-v2/releases/tag/v2.0.0). The [public Render service](https://pediatric-eeg-seizure-v2.onrender.com/) passed real-artifact API and browser verification on 2026-10-02.

[Try the public demo](https://pediatric-eeg-seizure-v2.onrender.com/) · [Verified release](https://github.com/iampenuel/pediatric-eeg-seizure-v2/releases/tag/v2.0.0) · [Open the full-cohort Colab workflow](https://colab.research.google.com/github/iampenuel/pediatric-eeg-seizure-v2/blob/main/notebooks/colab_full_cohort.ipynb) · [Original V1](https://github.com/iampenuel/pediatric-seizure-cnn) · [PhysioNet source](https://physionet.org/content/chbmit/1.0.0/)

## What changed from V1

V1 deployed a Keras CNN over processed 8-second windows using Streamlit. Its public repository does not include the training implementation or a verifiable patient split. V2 rebuilds its conceptual architecture under stronger controls; the historical V1 accuracy is not treated as a comparable result.

V2 preserves provenance from the source recording to each window, separates individuals across train/validation/test, fits normalization on training data only, and freezes model selection and thresholds before testing. The new interface uses FastAPI and a small static frontend, with ONNX Runtime inference.

## Frozen protocol

| Partition | Cases | Unique individuals |
|---|---|---:|
| Train | chb02, chb03, chb06, chb07, chb10–chb18, chb20, chb24 | 15 |
| Validation | chb05, chb08, chb19, chb23 | 4 |
| Test | chb01, chb21, chb04, chb09, chb22 | 4 |

**chb01 and chb21 are the same individual.** They remain together and contribute one patient-level metric row, alongside a separate case breakdown. This uses the full eligible cohort in one fixed split; it is not leave-one-patient-out cross-validation.

- 18 canonical bipolar EEG channels, 256 Hz, 8-second nonoverlapping windows.
- Label 1 means any positive overlap with an annotated seizure. Time intervals are half-open; incomplete trailing windows are omitted.
- No windows cross recording gaps. Seizure-free recordings and all eligible evaluation windows are retained.
- Original int16 samples and physical calibration are cached per recording. Windows are converted to microvolts and normalized on demand.
- Cache files store contiguous `(samples, 18)` int16 data; loaders return `(18, 2048)` windows. The cache layout is recorded in metadata and fingerprints. Legacy channel-major caches remain readable, but new preparation uses the contiguous layout to avoid scattered disk reads.
- Training-only channel mean/std, shared by both architectures. No extra filtering in P0.
- Class-weighted loss uses training counts. No class balancing of validation/test.
- Seed 42, Adam 0.001, batch 128, at most 20 epochs, early stopping after four epochs without improved validation patient-macro average precision.
- Each threshold maximizes validation patient-macro F1 on a 0.001 grid; ties select the higher threshold.
- The demo default is chosen by validation patient-macro AP, never by test performance.

## Verified frozen results

These held-out metrics were regenerated from the original frozen experiment after loss of the first evaluation artifacts. This was evaluation-only recovery, not a new model-selection run.

**Validation selection** (51,969 windows, 287 positive; four individuals):

| Model | Selected epoch | Validation patient-macro AP | Validation patient-macro F1 | Frozen threshold |
|---|---|---|---|---|
| baseline | 11 | 0.582622 | 0.359526 | 0.999 |
| residual | 13 | 0.572549 | 0.343287 | 0.927 |

The baseline remains the demo default because its validation patient-macro AP was higher. Training originally stopped after 15 baseline epochs and 17 residual epochs; the selected weights remain epochs 11 and 13.

**Final held-out test**: 147,727 windows, 210 positive and 147,517 negative; prevalence 0.142154%; 328.2822 evaluated hours. All eligible test windows were retained. Four canonical individuals are tested: chb01 (including chb21), chb04, chb09, chb22.

Pooled window metrics:

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC / AP |
|---|---|---|---|---|---|---|---|
| baseline | 0.988025 | 0.461905 | 0.988774 | 0.055334 | 0.098828 | 0.934473 | 0.159392 |
| residual | 0.950997 | 0.747619 | 0.951287 | 0.021381 | 0.041573 | 0.947571 | 0.404811 |

Confusion counts use `[[TN, FP], [FN, TP]]`: baseline `[[145861, 1656], [113, 97]]`; residual `[[140331, 7186], [53, 157]]`.

Patient-macro metrics (equal weight per individual, mean over defined values):

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC / AP |
|---|---|---|---|---|---|---|---|
| baseline | 0.987332 | 0.503481 | 0.988238 | 0.503859 | 0.296169 | 0.966449 | 0.568545 |
| residual | 0.943848 | 0.770959 | 0.944201 | 0.323532 | 0.344061 | 0.969320 | 0.607172 |

Every held-out individual:

| Model | Individual | Windows | Positive | Sensitivity | Specificity | Precision | F1 | AP |
|---|---|---|---|---|---|---|---|---|
| baseline | chb01 (+chb21) | 33,019 | 90 | 0.333333 | 0.999939 | 0.937500 | 0.491803 | 0.601530 |
| baseline | chb04 | 70,222 | 52 | 0.403846 | 0.994428 | 0.050971 | 0.090517 | 0.066370 |
| baseline | chb09 | 30,535 | 39 | 0.897436 | 0.958585 | 0.026965 | 0.052356 | 0.802225 |
| baseline | chb22 | 13,951 | 29 | 0.379310 | 1.000000 | 1.000000 | 0.550000 | 0.804055 |
| residual | chb01 (+chb21) | 33,019 | 90 | 0.677778 | 0.995991 | 0.316062 | 0.431095 | 0.594494 |
| residual | chb04 | 70,222 | 52 | 0.750000 | 0.990623 | 0.055954 | 0.104139 | 0.134768 |
| residual | chb09 | 30,535 | 39 | 0.897436 | 0.790333 | 0.005444 | 0.010823 | 0.868803 |
| residual | chb22 | 13,951 | 29 | 0.758621 | 0.999856 | 0.916667 | 0.830189 | 0.830623 |

Precision is low because the false-positive burden remains substantial, despite high aggregate accuracy. The residual model has higher test recall and AP but many more false positives, especially in chb09. The baseline misses all 27 positive chb21 windows at its frozen threshold; chb21 is a case breakdown of the same individual as chb01. No model, threshold, or default was changed after these observations. Event sensitivity, false alarms/hour, calibration, and external clinical validation were not evaluated.

The [versioned demo release](https://github.com/iampenuel/pediatric-eeg-seizure-v2/releases/tag/v2.0.0) contains full-precision pooled, patient-macro, per-patient, and per-case metrics in `manifest.json`, the frozen `study.json`, two ONNX models, frozen scalers, 29 attributed curated clips, and file hashes. Detailed ROC/PR data and figures, per-window predictions, original checkpoints, and recovery archives remain in the local research audit; they are not deployment assets and are not published in this release.

The 29 deduplicated held-out demo clips passed 58 model/example HTTP checks locally and on the public Render service. Public ONNX inference matched all 928 saved evaluation scores within `1.8342157753e-6`, with identical classifications, labels, thresholds, and source provenance. All 928 context-window scores agreed with saved PyTorch evaluation outputs within `1.8194589856e-6` (required `1e-5`), with identical classifications and labels. Independent CPU export checks on 32 validation windows per model had maximum errors `6.0355e-8` (baseline) and `1.5975e-7` (residual). The desktop and 390-pixel mobile interfaces were checked with both models and all four example categories. The frontend source is published at `59fa633eb368afe7991a69ec4571b25f8db0d5ba`; [clean Linux CI](https://github.com/iampenuel/pediatric-eeg-seizure-v2/actions/runs/37033458329) passed all 40 tests in 4.09 seconds and the lean serving HTTP smoke. The deployment-time local data/download/recovery subset passed 22 tests; browser and real-bundle checks are separate from unit tests.

Frozen experiment / evaluation code: `a20161cfecf112a874800c6a5bea4b314ccd6d71`. Baseline epoch 1 used `cbc5ff8c750df256a5618cba70c357e17bfb8ac5`; later training used the recorded I/O-only change with identical tensors. The experiment used Python 3.12.13, NumPy 2.0.2, PyEDFlib 0.1.42, PyTorch 2.11.0+cu128, scikit-learn 1.6.1, and a Tesla T4; complete package/hardware records are retained in the local audit.

| Frozen identifier | SHA-256 |
|---|---|
| Dataset | `a22541746f9adf856c807d970dfc84ee307a23559831613430cbd184c0cf8c33` |
| Window manifest | `7b8a1f122f6a93ce10c0124ff9732789246c030aabb402aa7349671c873988b9` |
| Split (canonical object) | `3feac96918a40011687f9a553c12be0f89d697fe18733bda2bd88c64b53f2cfc` |
| Shared scaler | `ae7d8f960c56b359fa0ecfb8425a95137b44375a2e37fba50450865e8fb10414` |
| Baseline checkpoint | `4ad0c5b57599f7a311125b1bb43e5fa5660fbd68e778a208d9171fde13b4e5ac` |
| Residual checkpoint | `bbe9bdd0186b87232f433cd4469f98c8335356fd6927d2d7609bbba830ad8c45` |
| Final demo archive | `53d0aedd5f73a0cf11fbb99d967c78706cd95461020ac77c51d273ac7d5fe7d5` |

For evaluation-only recovery, verify every recovery member first, restore into a new runtime directory, check out the recorded revision and matching environment, rebuild runtime-only caches, and require identical source/cache/metadata/window/split/config/scaler/checkpoint hashes and counts before `seizure-v2 evaluate`. The orchestration and equivalence receipt are preserved under `recovery/full-seed42/`. Do not run training, freezing, scaler fitting, or threshold selection to reproduce these reports.

## Verified local data quickstart

Use Python 3.12 on macOS or Linux. Raw data and generated artifacts are ignored by Git.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -e '.[data,test]'
seizure-v2 prepare --root data --recordings chb02/chb02_01.edf chb02/chb02_16.edf
```

The real-data check produces **569 windows: 558 non-seizure and 11 seizure-positive**. The seizure in `chb02_16.edf` spans 130–212 seconds; the positive windows span 128–216 seconds. These are data-pipeline counts, not model-performance results.

Run the same preparation command again to verify cache reuse. `reports/verification/` contains the small development manifests and verification record. Two additional validation recordings (`chb05_01` and `chb05_06`) were used only to test the training/reporting commands. No test individuals were used in local model development.

```bash
python -m pip install --no-cache-dir -e '.[data,train,export,serve,test]'
python -m pytest -q
seizure-v2 prepare --root data --recordings chb05/chb05_01.edf chb05/chb05_06.edf --mirror s3 --workers 2 --evict-raw
seizure-v2 train --root data --config configs/baseline.yaml --output runs/development/baseline --development --epochs 1 --num-workers 0 --device cpu
seizure-v2 train --root data --config configs/improved.yaml --output runs/development/residual --development --epochs 1 --num-workers 0 --device cpu
seizure-v2 freeze --runs runs/development/baseline runs/development/residual --output runs/development/study.json --development
seizure-v2 evaluate --root data --study runs/development/study.json --output runs/development/reports --validation-only --device cpu
```

Development artifacts cannot produce final test reports or a publishable demo bundle. Add `--resume` to the identical training command after interruption; a completed frozen run returns its verified original result. Use new output directories for a new experiment. The local dependency snapshot is `requirements-lock.txt`; actual GPU environment versions are recorded separately. On cloud-synced macOS folders, keep the virtual environment outside the synced folder to avoid filesystem offloading delays.

The full-cohort notebook uses bounded parallel downloads from PhysioNet's official S3 mirror and evicts downloader-owned raw staging files after verified conversion. Raw EEG, caches, and expanded demo clips stay on the **Colab runtime disk**. No Google Drive mount or access is used. The full-cohort preparation, training, frozen selection, evaluation, export, local recovery, and real inference checks have passed. The retained recovery archives and final demo are under the gitignored local `recovery/full-seed42/` folder.

In Colab, run the notebook's `run_with_downloads()` wrapper. It launches `python scripts/run_study.py --data /content/chbmit-v2 --output /content/eeg-v2-run/full-seed42 --interactive-backups`. At every completed epoch and natural checkpoint it pauses, offers a compact recovery download, and requires the locally verified SHA-256 before advancing. Backups contain checkpoint/optimizer/RNG state, frozen preprocessing/configuration, compressed window manifests, per-recording calibration/channel mappings, official source inventories and summaries, environment metadata, and reports; source signals and large arrays are excluded. The final demo archive is downloaded separately.

If the notebook's `download` prompt stalls, download `/content/eeg-v2-run/recovery-staging/eeg-recovery.tar.gz` through Colab's Files pane (right-click the file, then Download). This route was verified with the full preparation archive. Verify the downloaded archive locally, then paste its SHA-256 directly into the paused notebook prompt. Do not acknowledge a download that has not arrived and passed verification.

When resuming within the same intact runtime, `run_with_downloads(DATA, OUTPUT, reuse_prepared=True)` avoids repeating preparation but still hashes and audits the complete cache before training. This option fails if the full cache is missing or incomplete; after a runtime reset, use the default preparation workflow. Contiguous caches use normal operating-system read-ahead. Legacy channel-major caches use the random-access memory-map hint when supported. A training-only I/O comparison found identical tensors and faster contiguous-window reads with normal read-ahead; sample values and ordering are unchanged. Resumed runs retain the code/environment provenance of each execution segment.

Save downloads under the gitignored local `recovery/` directory. Verify each with `python -m seizure_v2.recovery verify --archive recovery/ARCHIVE --sha256 SHA`. To automatically retain only two verified snapshots and remove the temporary download copy, use `python scripts/receive_recovery.py DOWNLOAD --sha256 SHA --move`. Keep all persistent copies under `recovery/` to limit local storage. After a runtime reset, upload the latest archive through Colab's Files pane, check out its recorded Git commit, install dependencies, then run `python -m seizure_v2.recovery restore --archive /content/ARCHIVE --sha256 SHA --output /content/eeg-v2-run/full-seed42`. Restore only accepts a new destination. Rerunning the workflow regenerates runtime caches, checks dataset/configuration identity, and resumes the saved epoch; it never silently reduces the cohort or overwrites a frozen study.

## Source audit and exclusions

The live [RECORDS](https://physionet.org/files/chbmit/1.0.0/RECORDS) and [RECORDS-WITH-SEIZURES](https://physionet.org/files/chbmit/1.0.0/RECORDS-WITH-SEIZURES) inventories list 686 recordings and 141 seizure-containing recordings, including chb24. Older descriptive counts on the dataset page omit the later case.

One inventory error was found during verification: `RECORDS-WITH-SEIZURES` lists `chb07_18.edf`, but the [chb07 summary](https://physionet.org/files/chbmit/1.0.0/chb07/chb07-summary.txt) gives zero seizures there and a seizure at 13688–13831 seconds in `chb07_19.edf`. The official [checksum manifest](https://physionet.org/files/chbmit/1.0.0/SHA256SUMS.txt) lists the seizure sidecar for `_19`, not `_18`. The pipeline logs this narrow correction and fails on other inconsistencies.

chb24 omits non-seizure recordings from its summary; only those verified negative entries are recovered from the inventories. The three unsupported montages, chb12_27, chb12_28, and chb12_29, exclude 10,824 seconds and 13 annotated seizures. Missing channels are never interpolated or filled with zeros.

The verified source audit accounts for all 686 recordings: **683 eligible recordings** and the three disclosed exclusions. The verified manifest contains 241,232 training windows (1,062 positive), 51,969 validation windows (287 positive), and 147,727 test windows (210 positive). These are preparation counts, not performance results. `reports/full-cohort/` includes the source accounting, exclusions, patient window counts, and preparation audit. Its historical training-progress snapshot is not the final result; use the verified results above and the versioned release metrics for the completed study. The compressed complete window manifest and calibration metadata are retained in the gitignored recovery archive. The 15/4/4 canonical individual split is disjoint, and the full manifest again confirms the required 11 positive windows in chb02_16.

## Implementation map

| Area | Responsibility |
|---|---|
| `src/seizure_v2/data` | Checksummed downloads, annotation audit, channel mapping, int16 cache, windows, split, normalization |
| `src/seizure_v2/models` | Verified V1 topology and compact residual 1D CNN |
| `src/seizure_v2/training` | Deterministic seeds, training, resume, validation selection, study freeze |
| `src/seizure_v2/evaluation` | Pooled/patient/case metrics, ROC/PR, confusion matrices, failure provenance and waveforms |
| `src/seizure_v2/inference` | ONNX parity verification, release packaging, CPU inference |
| `web` | FastAPI service and responsive research explorer |
| `scripts/run_study.py` | Restartable full-cohort orchestration used by Colab |
| `tests` | Scientific guardrails, source/EDF checks, model/export shapes, unavailable-artifact behavior |

Each final run records the exact split/data hashes, source checksums, seed, configuration, checkpoint, scaler, selected threshold, package versions, and code revision. `best.pt` is portable model state; `last.pt` contains trusted local optimizer/RNG state for resuming. Never load an untrusted resume checkpoint.

## Reading results responsibly

Reports distinguish validation and test. They include accuracy, sensitivity, specificity, precision, F1, AUROC, average precision (AUPRC/AP), confusion matrices, ROC/PR curves, evaluated duration, prevalence, patient-macro metrics, and separate per-patient/per-case tables. Undefined metrics are JSON `null`.

False-positive and false-negative examples retain recording/time, true label, model score, prediction, and threshold. Plots document visual evidence; the code does not fabricate clinical explanations. Curated demo examples are illustrative, not a representative sample for estimating error frequency.

This is retrospective window classification. Four test individuals do not establish broad population generalization. Window sensitivity is not event-level sensitivity. An uncalibrated sigmoid output is a **seizure score**, not a true probability.

## Deployment

The deployment bundle is published at [GitHub Release v2.0.0](https://github.com/iampenuel/pediatric-eeg-seizure-v2/releases/tag/v2.0.0). It contains 36 files (33,919,613 archive bytes), with only the two ONNX models, frozen scalers, study/metrics metadata, attributed curated clips, and integrity manifest. Raw EDFs, signal caches, recovery archives, and training checkpoints are excluded.

Install the verified release and serve it through FastAPI so CSS, JavaScript, and API routes share the same origin:

```bash
python -m pip install --no-cache-dir -e '.[serve]'
export SEIZURE_ARTIFACT_URL=https://github.com/iampenuel/pediatric-eeg-seizure-v2/releases/download/v2.0.0/demo.tar.gz
export SEIZURE_ARTIFACT_SHA256=53d0aedd5f73a0cf11fbb99d967c78706cd95461020ac77c51d273ac7d5fe7d5
export SEIZURE_BUNDLE=artifacts/demo
python scripts/fetch_artifacts.py
uvicorn web.backend.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. Never open `index.html` as a standalone file. The build downloader verifies the archive SHA-256; the inference service verifies every manifest-listed file and rejects development artifacts. Missing or invalid artifacts produce an explicit unavailable state, never synthetic results.

`render.yaml` defines one free Python 3.12.10 service with a single Uvicorn worker. Configure the same three `SEIZURE_*` environment variables above. The build installs only serving dependencies and fetches the versioned bundle; it does not train or fit preprocessing. `/healthz` returns ready only after verification, with study SHA-256 `5c8ce0a4b71eb22bfc81c093d111d555c044706203b43756d0b935dc3a8ecafd`. `/api/research` exposes audited results and frozen thresholds; example routes perform actual CPU ONNX inference.

The [public Render service](https://pediatric-eeg-seizure-v2.onrender.com/) passed real-artifact API and browser verification on 2026-10-02: ready health check, exact audited metrics, both models, all four curated categories, 18-channel traces, source provenance, channel controls, timeline navigation, desktop layout, and 390-pixel mobile layout without page overflow. The public release download was independently streamed and hash-verified after upload. Render’s build log confirmed verified artifact installation. No scientific decision changed during deployment. Free Render services sleep when idle and can require approximately a minute to start again.

## Attribution and intended use

Guttag, J. (2010). **CHB-MIT Scalp EEG Database**, version 1.0.0. PhysioNet. [DOI: 10.13026/C2K01R](https://doi.org/10.13026/C2K01R). Source data are available under the Open Data Commons Attribution License v1.0. Also cite Ali Shoeb's 2009 MIT thesis, *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*, when using this dataset in research.

**Research/educational prototype. Not a diagnostic system. Not for clinical decision-making.**
