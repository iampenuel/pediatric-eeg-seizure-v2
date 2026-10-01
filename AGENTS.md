# Scientific and engineering guardrails

- Never mix canonical individuals across training, validation, and test. Treat chb01 and chb21 as one individual.
- Never fit preprocessing on validation/test, or select thresholds, checkpoints, architecture, or hyperparameters using test results.
- Preserve case, individual, recording, sample/time boundaries, labels, and source hashes for every window.
- Never fabricate results, hide exclusions, or describe uncalibrated sigmoid scores as probabilities.
- Final experiments use all 23 eligible individuals. Small subsets are for development only.
- The demo must reflect real model outputs and state that it is an educational research prototype, not for diagnosis or clinical decisions.
- Prefer simple, testable implementations. Complete P0 before optional features; no model zoo.
- Keep raw EEG, caches, checkpoints, credentials, and environments out of Git. Use versioned release artifacts for trained models and attributed demo clips.
- Run meaningful offline tests and explicit real-data integration checks. Document only commands actually exercised, with honest qualification of smoke versus full runs.
- Do not alter the frozen protocol after inspecting held-out test failures.
