# examples

Runnable demonstrations, one per capability. Not published, and nothing else depends on it.

```bash
./gradlew :examples:run                       # prints the list
./gradlew :examples:run --args="recipes"      # run one
```

Everything here is meant to be read as much as run. Several files are written to be copied into your
own codebase — `HttpEmbeddingVectorizer` in particular.

## Start with these

| Example | Shows |
|---|---|
| `recipes` | **Multi-label classification end to end.** A model distilled from a JSON keyword ruleset: grouped against ungrouped cross-validation, a threshold sweep, per-label thresholds, an explained prediction, and a hand-written held-out set using wording the rules never mention |
| `embedding-service` | Embeddings from LM Studio or any OpenAI-compatible server, compared against the hashing baseline. Prints the setup if no server is reachable |
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
| [`HttpEmbeddingVectorizer`](src/main/kotlin/io/skein/examples/embedding/HttpEmbeddingVectorizer.kt) | A `Vectorizer` over any OpenAI-compatible embeddings endpoint. ~120 lines on the JDK HTTP client, [tested against a real server](src/test/kotlin/io/skein/examples/embedding/HttpEmbeddingVectorizerTest.kt). Copy it and add your auth headers |
| [`EmbeddingServiceConfig`](src/main/kotlin/io/skein/examples/embedding/EmbeddingServiceConfig.kt) | The settings that have to be pinned for a remote model to be trustworthy |
| [`RecipeRuleset`](src/main/kotlin/io/skein/examples/recipes/RecipeRuleset.kt) | A rule interpreter in a dozen lines, making the point that the library never sees your rule format |

## Documentation

[All documentation](../docs/README.md) · [Getting started](../docs/getting-started.md) ·
[Embeddings](../docs/embeddings/README.md)
