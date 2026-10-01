# Pediatric EEG · Research Explorer

**How well can a compact seizure detector generalize to a patient it has never encountered?**

V2 studies that question using original CHB-MIT EDF recordings, a frozen patient-independent split, a V1-style CNN, and one compact residual CNN. A browser demo makes both successful predictions and failure examples inspectable.

**Status:** the data pipeline, both models, validation selection, reports, export, and API are implemented. Local tests and clean Linux CI pass. Real-data development runs exercised preparation, both architectures, freezing, validation reports, and frozen-run resume. Full-cohort GPU training, held-out research results, and the final deployed inference demo are pending. There are no invented final metrics.

[Open the full-cohort Colab workflow](https://colab.research.google.com/github/iampenuel/pediatric-eeg-seizure-v2/blob/main/notebooks/colab_full_cohort.ipynb) · [Original V1](https://github.com/iampenuel/pediatric-seizure-cnn) · [PhysioNet source](https://physionet.org/content/chbmit/1.0.0/)

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
- Training-only channel mean/std, shared by both architectures. No extra filtering in P0.
- Class-weighted loss uses training counts. No class balancing of validation/test.
- Seed 42, Adam 0.001, batch 128, at most 20 epochs, early stopping after four epochs without improved validation patient-macro average precision.
- Each threshold maximizes validation patient-macro F1 on a 0.001 grid; ties select the higher threshold.
- The demo default is chosen by validation patient-macro AP, never by test performance.

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

The full-cohort notebook uses bounded parallel downloads from PhysioNet's official S3 mirror and evicts downloader-owned raw staging files after verified conversion. Raw EEG, caches, and expanded demo clips stay on the **Colab runtime disk**. No Google Drive mount or access is used. Full-cohort execution is not yet claimed as verified.

In Colab, run the notebook's `run_with_downloads()` wrapper. It launches `python scripts/run_study.py --data /content/chbmit-v2 --output /content/eeg-v2-run/full-seed42 --interactive-backups`. At every completed epoch and natural checkpoint it pauses, offers a compact recovery download, and requires the locally verified SHA-256 before advancing. Backups contain checkpoint/optimizer/RNG state, frozen preprocessing/configuration, compressed manifests, environment metadata, and reports; source signals and large arrays are excluded. The final demo archive is downloaded separately.

Save downloads under the gitignored local `recovery/` directory. Verify each with `python -m seizure_v2.recovery verify --archive recovery/ARCHIVE --sha256 SHA`. Keep the latest two verified snapshots to limit local storage. After a runtime reset, upload the latest archive through Colab's Files pane, check out its recorded Git commit, install dependencies, then run `python -m seizure_v2.recovery restore --archive /content/ARCHIVE --sha256 SHA --output /content/eeg-v2-run/full-seed42`. Restore only accepts a new destination. Rerunning the workflow regenerates runtime caches, checks dataset/configuration identity, and resumes the saved epoch; it never silently reduces the cohort or overwrites a frozen study.

## Source audit and exclusions

The live [RECORDS](https://physionet.org/files/chbmit/1.0.0/RECORDS) and [RECORDS-WITH-SEIZURES](https://physionet.org/files/chbmit/1.0.0/RECORDS-WITH-SEIZURES) inventories list 686 recordings and 141 seizure-containing recordings, including chb24. Older descriptive counts on the dataset page omit the later case.

One inventory error was found during verification: `RECORDS-WITH-SEIZURES` lists `chb07_18.edf`, but the [chb07 summary](https://physionet.org/files/chbmit/1.0.0/chb07/chb07-summary.txt) gives zero seizures there and a seizure at 13688–13831 seconds in `chb07_19.edf`. The official [checksum manifest](https://physionet.org/files/chbmit/1.0.0/SHA256SUMS.txt) lists the seizure sidecar for `_19`, not `_18`. The pipeline logs this narrow correction and fails on other inconsistencies.

chb24 omits non-seizure recordings from its summary; only those verified negative entries are recovered from the inventories. Unsupported montages are excluded explicitly, including chb12_27, chb12_28, and chb12_29. The final manifest will report their duration and seizure counts. Missing channels are never interpolated or filled with zeros.

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

`render.yaml` describes a single free Python service. The service can show an honest unavailable-artifact interface before research results exist. The final demo requires a verified, full-cohort release bundle. Development artifacts are rejected.

Release artifacts contain ONNX models, frozen scalers/thresholds, held-out reports, attributed EEG clips, and hashes. Set `SEIZURE_ARTIFACT_URL` and `SEIZURE_ARTIFACT_SHA256` for build-time retrieval. `/healthz` returns 503 until models are verified; `/api/research` remains available to describe readiness. Free Render services sleep and may have approximately minute-long cold starts.

## Attribution and intended use

Guttag, J. (2010). **CHB-MIT Scalp EEG Database**, version 1.0.0. PhysioNet. [DOI: 10.13026/C2K01R](https://doi.org/10.13026/C2K01R). Source data are available under the Open Data Commons Attribution License v1.0. Also cite Ali Shoeb's 2009 MIT thesis, *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*, when using this dataset in research.

**Research/educational prototype. Not a diagnostic system. Not for clinical decision-making.**
