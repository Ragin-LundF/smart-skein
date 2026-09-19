package io.skein.classify.spi

import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.domain.Explanation
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabelPrediction
import io.skein.classify.domain.MultiLabelPredictionFactory

/**
 * Port for a model that assigns **any number** of labels to a record, including none.
 *
 * A sibling of [Classifier], not a replacement for it. The two answer different questions and a
 * caller picks one by asking whether two labels can be true at the same time:
 *
 * - No — the record has exactly one category. Use [Classifier]: its softmax makes the labels
 *   compete, which is the correct model of a mutually exclusive choice.
 * - Yes — labels co-occur, or none may apply. Use this port: every label gets its own independent
 *   one-vs-rest decision, so `INSURANCE` and `LIFEINSURANCE` can both fire, and a record that
 *   matches nothing can come back empty.
 *
 * Forcing a co-occurring taxonomy through a softmax is not a rounding error: the scores are
 * constrained to sum to one, so a genuine second label is suppressed by construction.
 *
 * Fitting lives in [BatchLearner], not here — the optimiser these models want needs the whole
 * corpus at once and has no meaningful "learn one observation" step. An implementation of this port
 * is therefore an already-trained model.
 */
interface MultiLabelClassifier {

    /**
     * Raw per-label scores in logit space, before any sigmoid. This is the input to
     * [MultiLabelPredictionFactory.fromLogits] and to per-label calibration.
     *
     * Unlike [Classifier.logScores] these are **not** shift-invariant: each label's score is an
     * absolute margin, and adding a constant to all of them changes every probability. An
     * implementation must return the real margins.
     */
    fun logits(features: FeatureVector): Map<Label, Double>

    /**
     * Independent per-label probabilities in `[0, 1]`. They do **not** sum to one, and reading them
     * as a distribution is the standard way to misuse multi-label output.
     *
     * The default applies the overflow-safe logistic function to [logits], which is correct for
     * every one-vs-rest model; override only to apply a different link function.
     */
    fun scoreAll(features: FeatureVector): Map<Label, Double> {
        return logits(features = features)
            .mapValues { entry -> MultiLabelPredictionFactory.logistic(logit = entry.value) }
    }

    /** Scores [features] and ranks the result, accepting labels at or above [threshold]. */
    fun predict(features: FeatureVector, threshold: Double): MultiLabelPrediction {
        return MultiLabelPredictionFactory.fromLogits(logits = logits(features = features), threshold = threshold)
    }

    /** Labels this model was fitted for. */
    fun labels(): Set<Label>

    /**
     * Ranked per-feature contributions to [label]'s logit, largest absolute contribution first and
     * truncated to [limit] entries. Returns `null` when this implementation cannot attribute its
     * score to individual features, which is the default.
     *
     * Because each label is an independent linear head, the decomposition here is exact and needs
     * no centering: the contributions plus the intercept sum to the label's logit.
     */
    fun explain(features: FeatureVector, label: Label, limit: Int): Explanation? {
        return null
    }

    /** The tuning this model was fitted with, so it can be persisted and restored alongside it. */
    fun hyperparameters(): ClassifierHyperparameters {
        return ClassifierHyperparameters.DEFAULTS
    }
}
