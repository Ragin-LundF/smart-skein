# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-09-07

### Added

- **Model evaluation** (`skein-classify`) — `ModelEvaluator` scores a trained classifier against
  held-out data, or trains fresh models over a stratified holdout or k folds and scores those.
  Reports accuracy, per-class precision/recall/F1/support, macro/micro/weighted averages, a
  confusion matrix, top-k accuracy, log loss, Brier score, and reliability bins with expected
  calibration error. `StratifiedSplitter` produces deterministic, label-balanced splits that always
  leave every label at least one training example.
- **`skein-cli evaluate`** — prints an evaluation report from a saved model, either against a fresh
  labeled CSV or by retraining over the model's stored corpus, with per-label and confusion-matrix
  CSV output and a `--min-accuracy` gate that exits with code 2 for use in CI.
- **Probability calibration** (`skein-classify`) — `TemperatureCalibrator` fits a temperature on
  held-out data by minimizing negative log-likelihood; `ClassificationService.calibration` applies
  it to every prediction. Calibration is rank-preserving, so it never changes which label wins, only
  how confident the model claims to be. Naive Bayes probabilities are overconfident by construction,
  which is what this exists to correct.
- **Abstention** — `Prediction.isConfident`, `ClassificationService.classifyOrNull`, and
  `skein-cli predict --min-confidence`. An abstained row is written with an empty label cell and its
  confidence intact, so it flows straight back into `skein-cli label` as a pending row.
- **Explanations** (`skein-classify`) — `ClassificationService.explain` and `Classifier.explain`
  return the ranked per-feature contributions behind a prediction, as an exact additive
  decomposition of the score that drives the probability. Contributions are mean-centered, so a
  feature equally likely under every label contributes zero. Buckets stay opaque by default;
  `AttributionModeEnum.WITH_NGRAMS` resolves them to source fragments, re-derived on the call with
  no stored index.
- **Classifier hyperparameters are persisted** — `ClassifierHyperparameters` records a classifier's
  tuning, `Classifier.hyperparameters()` reports it, and `ModelStore` stores it so a tuned model is
  restored as the same model. `ClassifierFactory.create(kind, hyperparameters)` rebuilds it.
- **CRF model persistence** (`skein-extract`) — `CrfModelStore` saves and loads a trained
  `CrfSequenceLabeler` as a single versioned `SKCR` file, with no new dependencies. Hyperparameters
  and the SGD step counter are persisted, so training resumes exactly where it stopped rather than
  silently restarting with default settings. `CrfModelStore.metadata` reads the header without
  inflating the weights.

### Changed

- `ClassificationService` exposes `schema`, `classifier` and `featureStore` as read-only properties
  (they were already public constructor parameters). Source- and binary-compatible.
- `Classifier` gains `logScores` and `explain`, both with default implementations, so existing
  third-party classifiers keep compiling and linking unchanged. Implementations that override
  `classify` but not `logScores` fall back to `ln(probability)`, which underflows on confident
  models — **third-party classifiers should override `logScores`.**
- `ModelStore.load` now reports a truncated or corrupt file as an `IllegalArgumentException` instead
  of leaking a raw `EOFException`.
- `Classifier` also gains `hyperparameters()`, with a default reporting the library values, so
  existing implementations keep compiling and linking.

### Security

- A saved CRF model contains fragments of the training text in clear, because a CRF learns features
  keyed by the token text itself. This is unlike classification, whose features are irreversible
  hashes. `CrfModelStore` therefore defaults to `FeatureRetentionEnum.FREQUENT_ONLY`, dropping
  lexical features seen fewer than twice. That **reduces** exposure without eliminating it — only
  `FeatureRetentionEnum.STRUCTURAL_ONLY` carries a zero-clear-text guarantee, and only
  `ALL_FEATURES` restores a bit-identical model.

### Fixed

- **A tuned model no longer changes when saved and reloaded.** `ModelStore` recorded no classifier
  hyperparameters, and loading replays the stored observations through a freshly built classifier —
  so a logistic-regression model trained with, say, `initialLearningRate = 0.9, decayRate = 0.5,
  l2Regularization = 0.3` came back with the library defaults and predicted differently (0.7397
  versus 0.6731 on the regression case now covered by a test).
- **`HashingVectorizer` no longer throws on long input.** Its UTF-8 encode buffer and word-boundary
  buffer were fixed 256-element arrays, so a record with more than 128 words, or a word n-gram
  exceeding 256 UTF-8 bytes, raised `ArrayIndexOutOfBoundsException`. Both now grow on demand;
  an outlier record uses a one-off buffer rather than permanently inflating per-thread memory. The
  capacity check is per call, not per n-gram, so the hot path is unchanged.

### Breaking

- **Model file format.** `.skein` files are now written at version 2 (the calibration temperature was
  appended). Version 2 readers accept both v1 and v2 files, but **a 1.1.0 reader rejects a 1.2.0
  file**, because its version check is an exact match.
- `LoadedModel` gains `calibration` and `hyperparameters`. Its getters and `component1..4` are
  unchanged, so reading a loaded model is unaffected, but the four-argument constructor and the old
  `copy` signature are gone.
- `skein-cli export` writes a `skein-model 2` header with an added `calibration` line.

## [1.1.0] - 2026-06-28

- Fixed some performance issues in training for classification
- Added a new example, `logwatch`, for monitoring log files and extracting structured data with classification

## [1.0.0] - 2026-06-28

### First final release

- **Classification** — Naive Bayes and logistic regression classifiers for assigning labels to text records. Supports active learning so the model improves over time as more labeled data is provided.
- **Privacy-preserving features** — personal data never leaves the system in plain text. Feature extraction uses one-way hashing, so the original content cannot be reconstructed from a saved model.
- **Text foundation** — shared text normalization, broken-word repair (e.g. `"apart ment" → "apartment"`), and a typed tokenizer that turns free text into pattern signatures.
- **Extraction** — pull structured values from text using typed-token patterns and slot filling.
- **PostgreSQL storage** — optional persistence adapter for storing labeled observations in a database, with AES-256-GCM encryption at rest.
- **CLI tools** — command-line interface for interactive data labeling (`label`), batch classification (`predict`), and model export (`export`). Supports CSV input and output with configurable delimiters.
- **Duplicate filtering** — repeated ingestion of the same record is detected and silently ignored, keeping the model and saved file size stable.
- **Model file format** — trained models are saved to a single `.skein` file and can be restored without retraining from source data.
- **BOM** — a Bill of Materials module is provided for consumers who want to align all Skein module versions in one place.
