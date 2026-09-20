# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-09-20

### Added

- **Multi-label classification** (`skein-classify`) — `MultiLabelClassifier` and `BatchLearner`, two
  new SPI ports beside `Classifier`, for taxonomies whose labels co-occur. Every label gets an
  independent one-vs-rest decision, so a record can carry several labels or none; a softmax
  constrains its scores to sum to one, which suppresses a genuine second label by construction.
  `Classifier` is unchanged — a caller picks a port by asking whether two labels can be true at once.
  Recorded in [ADR 0001](docs/adr/0001-multi-label-classification.md), which also widens the module's
  documented responsibility.
- **L-BFGS batch training** (`skein-classify`) — `LbfgsMinimizer` (two-loop recursion, initial
  Hessian scaling, Armijo backtracking, curvature-pair damping) and `LogisticObjective`, an
  L2-regularised logistic loss over a `SparseMatrix` using scikit-learn's conventions exactly: `C`
  multiplies the data term and the intercept is not regularised. `LbfgsMultiLabelLearner` fits one
  head per label over a shared design matrix. Complementary to the existing online SGD classifier
  rather than a replacement — SGD for incremental and active learning, L-BFGS when the corpus is in
  hand and the best fit is wanted.
- **Grouped splitting** (`skein-classify`) — `GroupedSplitter` and `MultiLabelCrossValidator` keep
  rows sharing an origin inside one fold. Without it, rule-derived training data leaks across folds
  and every score is inflated: measured micro-F1 0.79 random against 0.75 grouped, macro-F1 0.634
  against 0.534. Cross-validation takes `MultiLabeledText` and builds a vectorizer per fold from that
  fold's training rows alone, so anything fitted to the corpus cannot leak into the features.
- **Weight persistence** (`skein-classify`) — `.skein` format version `0x03` stores a fitted weight
  matrix rather than a replayable corpus. Weights are pruned by magnitude (`keepFraction`), stored
  feature-major, delta-coded per row and written as `float16` where the range allows, with an
  automatic fallback to `float32`. `ModelStore.saveMultiLabel` / `loadMultiLabel`, plus
  `ClassifierKindEnum.MULTI_LABEL_LOGISTIC`.
- **Vectorizer identity** (`skein-classify`) — the `Vectorizer` port, `VectorizerFingerprint`, and a
  fingerprint check on load that throws `VectorizerMismatchException` on a mismatch. Scoring a model
  with a different featurisation does not fail on its own; it returns confident, wrong labels
  silently, which is why this is fatal rather than a warning. `HashingVectorizer` implements the port
  and fingerprints its key, width, n-gram ranges, term weighting and normalizer.
- **Scale hardening** (`skein-classify`) — `SparseMatrix` blocks rows so no array approaches
  `Int.MAX_VALUE` (~8.5M documents at measured density) and stores values as `float`;
  `TrainingParallelism` bounds concurrent fits from heap and feature count rather than core count;
  `TokenMasker` collapses high-cardinality tokens via `skein-text`'s `TypedTokenizer` (measured 59%
  fewer features, 60% smaller model); `CorpusDeduplicator` collapses rows identical once masked
  (measured 400,000 documents to ~21,000) and reports the ratio.
- **Optional refinements** (`skein-classify`) — `TermWeightingEnum.SUBLINEAR` for `1 + ln(count)`
  weighting; `IdfVectorizer` and `DocumentFrequencyTable` for IDF over hashed buckets, with a
  document-frequency floor that scales with the corpus and a fitted table persisted alongside the
  model; `LabelThresholds` and `ThresholdOptimizer` for a per-label acceptance threshold, since a
  global cut silences a rare label while a well-supported one over-fires at the same value.
- **Recipe tagging example** (`examples`) — `./gradlew :examples:run --args="recipes"`. A multi-label
  model distilled from a JSON keyword ruleset, with grouped and ungrouped cross-validation side by
  side, a threshold sweep, per-label thresholds, an explained prediction, and a hand-written held-out
  set using wording the rules never mention — the only number that says whether the model generalises
  past the rules.
- **`skein-classify-embedding-onnx`** — a new published adapter module implementing `Vectorizer`
  with an external ONNX sentence-embedding model, so a classifier can generalise past the literal
  wording hashed n-grams match. `OnnxEmbeddingVectorizer.open(modelPath, tokenizerPath)` wires ONNX
  Runtime and a Hugging Face tokenizer; mean or CLS pooling, optional L2 normalisation, batch
  embedding, and a bounded LRU `EmbeddingCache` so a hyperparameter sweep embeds each record once.
  Its fingerprint digests the **model file's contents**, the tokenizer file, the pooling strategy,
  the normalisation and the hidden size, so a model swapped in place under an unchanged name is
  refused at load. Dense vectors are expressed through the existing sparse `FeatureVector`, so the
  learner, the objective and the scoring loop are unchanged. The ONNX Runtime and tokenizer native
  binaries live only in this module — a consumer using feature hashing inherits neither.
- **`HttpEmbeddingVectorizer`** (`examples`) — a `Vectorizer` over any OpenAI-compatible
  `/embeddings` endpoint, so LM Studio, Ollama, llama.cpp, vLLM and the OpenAI API all work
  unchanged. Built on the JDK HTTP client with no dependency worth shipping, meant to be copied
  rather than consumed, and covered by tests against a real local HTTP server. Two runnable
  examples, `embedding-service` and `embedding-onnx`, train the recipe tagger on embeddings and
  score it against the hashing baseline; both print setup instructions when the service or model is
  absent.
- **`skein-classify-embedding-http`** — a new published adapter module implementing `Vectorizer`
  against any OpenAI-compatible `/embeddings` endpoint, so LM Studio, Ollama, llama.cpp, vLLM, Text
  Embeddings Inference and the OpenAI API all work unchanged. Promoted out of `examples`, where it
  was a copy-me file, because a supported route B needs a supported artifact. `EmbeddingTransport`
  isolates the HTTP call, so the protocol is this module's business and retries, proxies and
  connection pooling stay the caller's. `EmbeddingServiceConfig` gained `headers` for auth, kept
  out of the fingerprint so a rotated token does not invalidate a trained model, and redacted from
  `toString`. The only dependency beyond `skein-classify` is a JSON parser.
- **Vector canary** (`skein-classify`) — `VectorizerCanary`, `Vectorizer.canary()` and
  `VectorizerCanaryException`. Fixed probe texts are embedded when a model is saved and re-embedded
  when it is loaded; a model changed on the server moves them and the load throws. This is the only
  check that looks at what a vectorizer *does* rather than what it declares, and it is what makes
  route B usable in production: a fingerprint cannot see a service swapping its weights, because
  nothing in the protocol reveals them. Opt in by setting probe texts; a model saved without them
  behaves exactly as before. Drift is measured as relative L2 rather than cosine distance, so a
  service that starts L2-normalising its output — which rescales every vector without rotating it,
  and breaks a classifier trained on the unnormalised ones — is caught rather than waved through.
  `saveMultiLabel` re-checks the canary against the live vectorizer before writing, which catches a
  model swapped *during* a long training run. `loadMultiLabel` gained `verifyCanary` to skip it,
  and **performs network I/O when a canary is present**.
- **`BatchVectorizer`** (`skein-classify`) — a `Vectorizer` sub-port for implementations where
  featurising many texts at once is materially cheaper than one at a time, plus a
  `Vectorizer.vectorizeAll(texts)` extension that uses it when present and loops when it is not.
  Both embedding adapters implement it; `HashingVectorizer` deliberately does not, because hashing
  a record takes microseconds and there is nothing to amortise — which is what keeps the port a
  capability a caller can *detect* rather than a method everything claims. `MultiLabelCrossValidator`
  now featurises a fold in one call instead of one per row: with an embedding vectorizer that was
  one inference call, or one HTTP round trip, per row **per fold**.
- **Binary-compatibility validation** — every published module carries an ABI dump under
  `<module>/api/`, checked by `checkKotlinAbi` as part of `check` and re-recorded with
  `updateKotlinAbi`. Kotlin's own ABI validation rather than the standalone
  binary-compatibility-validator plugin, whose bundled ASM cannot read JDK 25 class files. The
  `HashingConfig` break below was found by reading a diff; this is so the next one is not.
- **[`docs/migration.md`](docs/migration.md)** — what a 1.2.0 consumer has to do to move to 2.0.0.
- **Restructured documentation** — every module carries a short `README.md` stating what it is and
  when to use it, linking into depth under [`docs/`](docs). New: an
  [index](docs/README.md), [getting started](docs/getting-started.md),
  [architecture](docs/architecture.md) with module, layer and pipeline diagrams, seven
  [classification](docs/classify/README.md) pages, and four
  [embedding](docs/embeddings/README.md) pages covering both integration routes, LM Studio setup
  and model selection. Diagrams are Mermaid with committed SVG exports and a render script in
  [`docs/assets/diagrams`](docs/assets/diagrams).
- **[`docs/bring-your-own-data.md`](docs/bring-your-own-data.md)** — six steps from your records to a
  model, including the single question that decides single- versus multi-label.

### Breaking

- **`HashingConfig` gained a `termWeighting` property.** It is appended with a default, so source
  using named or positional arguments keeps compiling, but the generated `copy` and `componentN`
  signatures change — a binary break for anything compiled against 1.2.0. **Recompiling is the
  whole fix.**
- **Model file format.** Multi-label models are written at `.skein` payload version `0x03`, which
  stores a fitted weight matrix rather than a replayable corpus. The version byte selects the
  payload *shape*, so **a 1.x reader rejects a v3 file** at the header rather than misreading it.
  Single-label v1 and v2 files are unaffected and load unchanged.
- **`LoadedMultiLabelModel` gained `canary`.** Appended last with a default, so positional and
  named construction keep compiling, but the constructor, `copy` and `componentN` signatures
  change.
- `ModelStore.loadMultiLabel` gained a defaulted `verifyCanary` parameter on both overloads.
  Source-compatible; the old signatures no longer exist at the bytecode level.
- `Vectorizer` gained `canary()` with a default implementation returning `null`, so existing
  implementations keep compiling and linking. `BatchVectorizer` is a new sub-interface rather than
  a method on `Vectorizer`, so nothing existing has to implement it.
- `OnnxEmbeddingVectorizer.vectorizeAll` and `HttpEmbeddingVectorizer.vectorizeAll` are now
  `override`s of `BatchVectorizer.vectorizeAll` rather than methods on the concrete class.
  Source-compatible; callers holding either concrete type are unaffected.
- **`HttpEmbeddingVectorizer` and `EmbeddingServiceConfig` moved out of `examples`** into
  `io.skein.classify.embedding.http.{infrastructure,domain}`, in the new published module. Nothing
  depended on `examples`, so this breaks only a copy someone took by hand — which was the
  documented way to use it, and is the reason it is now a module.

### Changed

- `.ai/instructions/module-architecture.md` — `skein-classify`'s responsibility is now "assigning
  labels to a whole record — one label, or several when labels co-occur".

### Fixed

- **Published POM metadata.** Every module published a POM whose `<name>` was just the artifact id
  and whose `<description>` was the placeholder "Skein Library", discarding the `publishName` and
  `publishDescription` each build file sets. The publishing convention is applied from a module's
  `plugins { }` block, which runs before the body that sets those extras, so reading them eagerly
  always saw nothing. They are read lazily now.
- `GroupedSplitter` orders equal-sized groups by a seeded mix of their names rather than
  alphabetically. Group names encode structure, so alphabetical order correlates with it and
  round-robin assignment can line a fold up with an entire category — five methods under five folds
  put every `one pot` group in one fold, leaving the label that depends on it absent from four folds'
  training data.
- `ThresholdOptimizer` evaluates every distinct score in a single descending sweep instead of
  sampling a fixed number of candidates. Sampling could step over the one threshold that separates
  the classes cleanly and settle for a worse cut, and the sweep is `O(n log n)` rather than `O(n^2)`.

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
