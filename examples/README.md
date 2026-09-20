# examples

Runnable demonstrations, one per capability. Not published, and nothing else depends on it.

```bash
./gradlew :examples:run                       # prints the list
./gradlew :examples:run --args="recipes"      # run one
```

Everything here is meant to be read as much as run.

## Start with these

| Example | Shows |
|---|---|
| `recipes` | **Multi-label classification end to end.** A model distilled from a JSON keyword ruleset: grouped against ungrouped cross-validation, a threshold sweep, per-label thresholds, an explained prediction, and a hand-written held-out set using wording the rules never mention |
| `embedding-service` | **Are embeddings worth it?** Discovers a model on LM Studio or any OpenAI-compatible server, trains on it, and scores it against the hashing baseline on wording the rules never mention |
| `localai` | **How do I run that safely?** Discover the model, calibrate the canary tolerance against it, then watch a model swap get caught. Falls back to an in-process stub server, so it runs with nothing installed |
| `embedding-onnx` | An embedding model run in-process through ONNX Runtime. Prints the export command if no model is configured |
| `transaction` | Classify → route → extract, the original end-to-end pipeline |

`recipes` is the one to read first if you are new. It is also the only example that shows the
measurement most people skip: how much worse a rule-distilled model does on wording the rules do not
cover.

## Everything else

**Foundations**

| | |
|---|---|
| `textrepair` | Frequency model, broken-word repair, the full normalize → tokenize pipeline |
| `patternmatching` | The TokenPattern DSL: `findAll`, `matchesFully`, partial and non-matching cases |
| `slots` | Positional against key-anchored extraction |
| `schemainference` | Infer a schema from samples, classify with it, and its edge cases |
| `validation` | Valid, invalid and warning-bearing records |
| `clustering` | Unsupervised layout discovery from a mixed corpus |
| `persistence` | `ModelStore` round-trip, frequency model serialisation |
| `import` | Streaming import with validation, feeding a classifier |

**Workflows**

| | |
|---|---|
| `regression` | Naive Bayes → logistic regression retraining, confidence comparison |
| `activelearning` | Uncertainty sampling, feedback, metrics |
| `crf` | Train a CRF token tagger, generalise to unseen input, save and resume |
| `explain` | Calibrate confidences, abstain, explain a prediction |
| `clidemo` | Schema inference plus an active-learning loop, with CLI equivalents |
| `tokenization` | `WHITESPACE` against `PUNCTUATION_AWARE` |
| `customtokens` | A custom `TokenPatternConfig` with a domain recogniser |

**Edge cases**

| | |
|---|---|
| `normalizeredges` | Normalizer idempotence and boundaries |
| `privacy` | PII exclusion from features |
| `signature` | Identical layouts produce identical fingerprints |
| `clitool` | Predict all rows and export a model as readable text |
| `logwatch` | Train an anomaly detector from a keyword rules CSV, then scan logs |

## The corpora

`recipes` and both embedding examples share one corpus, in
[`RecipeCorpus`](src/main/kotlin/io/skein/examples/recipes/RecipeCorpus.kt):

- **~400 generated recipes** from a seeded combinatorial template, labelled by running
  [`recipe-rules.json`](src/main/resources/recipe-rules.json). Committed as a generator with a fixed
  seed, so it is identical on every machine.
- **40 hand-written recipes** using wording the rules miss — `aubergine`, `courgette`,
  `cacio e pepe`, `sans gluten` — labelled by hand.

The second set is the point. A model trained only on rule-generated data largely learns the rules;
this is the only thing that measures whether it generalises past them.

## Copyable pieces

| File | What it is |
|---|---|
| [`RecipeRuleset`](src/main/kotlin/io/skein/examples/recipes/RecipeRuleset.kt) | A rule interpreter in a dozen lines, making the point that the library never sees your rule format |
| [`EmbeddingComparison`](src/main/kotlin/io/skein/examples/embedding/EmbeddingComparison.kt) | Scoring a hashing baseline against an embedding vectorizer on the same held-out set — the measurement that decides whether embeddings are worth it for you |
| [`ModelDiscovery`](src/main/kotlin/io/skein/examples/localai/ModelDiscovery.kt) | Finding a model on an OpenAI-compatible server that *actually embeds*. The id is not guessable and `/v1/models` lists what is downloaded rather than loaded |
| [`CanaryCalibrator`](src/main/kotlin/io/skein/examples/localai/CanaryCalibrator.kt) | Measuring your own model's drift so a canary tolerance is chosen from data, and reporting drift on a timer without failing the caller |
| [`StubEmbeddingServer`](src/main/kotlin/io/skein/examples/localai/StubEmbeddingServer.kt) | A development double, **not** for production: an in-process embeddings endpoint whose weights you can change mid-run, which is the only way to stage a model swap in a test |

`HttpEmbeddingVectorizer` used to live here as a copy-me file. It is now
[`skein-classify-embedding-http`](../skein-classify-embedding-http), a published module, because a
supported route B needs a supported artifact. The parts people actually changed in their copies are
configuration now: auth headers go in `EmbeddingServiceConfig.headers`, and anything about
retries, proxies or pooling belongs in your own `EmbeddingTransport`.

## Running the local-AI example

```bash
./gradlew :examples:run --args="localai"                                    # finds a server, or stubs one
./gradlew :examples:run --args="localai" -Dskein.localai.url=http://localhost:11434/v1   # Ollama
./gradlew :examples:run --args="localai" -Dskein.localai.model=<id>         # skip discovery
```

Every `-Dskein.*` property is forwarded to the example's JVM. With no server reachable it starts
[`StubEmbeddingServer`](src/main/kotlin/io/skein/examples/localai/StubEmbeddingServer.kt) on a
loopback port and runs the whole walkthrough against that — including the model swap, which a real
server cannot be asked to perform on demand.

[`LocalAiVerificationTest`](src/test/kotlin/io/skein/examples/localai/LocalAiVerificationTest.kt)
asserts the same behaviour the example prints, so the guarantees it teaches cannot quietly stop
being true. It runs in the ordinary build, with no service installed.

## Documentation

[All documentation](../docs/README.md) · [Getting started](../docs/getting-started.md) ·
[Embeddings](../docs/embeddings/README.md)
