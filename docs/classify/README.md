# skein-classify

Assigning labels to a whole record — one label, or several when labels co-occur.

Everything here is classical statistical machine learning on the CPU: no GPU, no neural network, no
external service. Features are irreversible keyed hashes by default, so personal data never enters a
model in clear text. Predictions come with calibrated confidences, an abstain option, and an exact
per-feature decomposition of the score.

## Pages

| | |
|---|---|
| [Schema and records](schema.md) | Describing your data, keeping PII out of features, bulk import |
| [Featurisation](featurisation.md) | Feature hashing, n-grams, the privacy guarantee, term weighting, IDF |
| [Single-label](single-label.md) | `Classifier`, Naive Bayes, online SGD, calibration, explanations, active learning |
| [Multi-label](multi-label.md) | `MultiLabelClassifier`, L-BFGS batch training, per-label thresholds |
| [Evaluation](evaluation.md) | Metrics, grouped cross-validation, threshold sweeps |
| [Persistence](persistence.md) | The `.skein` format, vectorizer fingerprints |
| [Scale](scale.md) | Above a million records: masking, deduplication, memory |

For semantic features instead of literal n-grams, see [Embeddings](../embeddings/README.md).

## The one decision that shapes everything

> **Can two labels be true of the same record at the same time?**

**No** — one category per record. Use `Classifier`. Its softmax makes labels compete, which is the
correct model of a mutually exclusive choice, and it learns one observation at a time so it supports
incremental and active learning.

**Yes** — labels co-occur, or none may apply. Use `MultiLabelClassifier` with `BatchLearner`. Every
label gets an independent one-vs-rest decision, so two can fire together and a record matching
nothing comes back empty.

Forcing co-occurring labels through a softmax is not a rounding error. The scores are constrained to
sum to one, so a genuine second label is suppressed by construction and no tuning recovers it.

| | `Classifier` | `MultiLabelClassifier` |
|---|---|---|
| Labels per record | exactly one | any number, including none |
| Scores | softmax, sum to 1 | independent sigmoids, do not sum to 1 |
| Learns | one observation at a time | the whole corpus at once |
| Incremental updates | yes | no — refit |
| Active learning | yes | not directly |
| Implementations | `NaiveBayesClassifier`, `LogisticRegressionSgdClassifier` | `MultiLabelLogisticClassifier` via `LbfgsMultiLabelLearner` |

The two are siblings, not alternatives ranked by quality. Pick by the question above.

## Package layout

| Package | Contains |
|---|---|
| `domain` | `Schema`, `Record`, `FeatureVector`, `Label`, `Prediction`, metrics, `SparseMatrix` |
| `spi` | `Classifier`, `MultiLabelClassifier`, `BatchLearner`, `Vectorizer`, `FeatureStore`, `RecordSource` |
| `application` | `ClassificationService`, `HashingVectorizer`, `ModelStore`, evaluators, splitters |
| `infrastructure` | The learning algorithms, L-BFGS, the weight matrix, hash maps |
