package io.skein.classify.application

import io.skein.classify.domain.MultiLabelMetrics
import io.skein.classify.domain.MultiLabelMetricsFactory
import io.skein.classify.domain.MultiLabelOutcome
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.ThresholdPoint
import io.skein.classify.spi.MultiLabelClassifier

/** Thresholds swept by [MultiLabelEvaluator.sweep] unless the caller supplies its own. */
private val DEFAULT_THRESHOLDS = listOf(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

/** Rank depths reported by [MultiLabelEvaluator.recallAtK] unless overridden. */
private val DEFAULT_RANK_DEPTHS = listOf(1, 2, 3, 5)

/**
 * Measures multi-label model quality: scores a fitted model against held-out data and reports what
 * it costs to move the acceptance threshold.
 *
 * The multi-label sibling of [ModelEvaluator], and deliberately narrower: it scores the model it is
 * handed and trains nothing. Cross-validation — which trains fresh models and therefore measures a
 * *recipe* rather than an artifact — lives in [MultiLabelCrossValidator], where the grouped split
 * that makes it honest also lives.
 */
class MultiLabelEvaluator {

    /**
     * Scores [classifier] against [holdout] at [threshold]. Trains nothing and mutates nothing, so
     * the same model can be scored against several holdouts.
     */
    fun evaluate(
        classifier: MultiLabelClassifier,
        holdout: List<MultiLabeledFeatures>,
        threshold: Double,
    ): MultiLabelMetrics {
        require(value = holdout.isNotEmpty()) { "cannot evaluate against an empty holdout" }
        return MultiLabelMetricsFactory.from(
            outcomes = outcomes(classifier = classifier, holdout = holdout, threshold = threshold),
        )
    }

    /**
     * Scores every case in [holdout] once, keeping the full ranking so a caller can re-read the
     * same scores at any threshold without re-running the model.
     */
    fun outcomes(
        classifier: MultiLabelClassifier,
        holdout: List<MultiLabeledFeatures>,
        threshold: Double,
    ): List<MultiLabelOutcome> {
        return holdout.map { observation ->
            MultiLabelOutcome(
                expected = observation.labels,
                prediction = classifier.predict(features = observation.features, threshold = threshold),
            )
        }
    }

    /**
     * The precision/recall/coverage table across [thresholds], ascending.
     *
     * This is the output a threshold decision should actually be made from, and the reason it is a
     * table rather than a single tuned number: there is no technically correct threshold. Accepting
     * more labels always trades precision for recall, and which side of that trade is right depends
     * on what a wrong label costs against what a missing one costs — a business question.
     *
     * Re-reads the scores already in [outcomes] rather than re-scoring, so sweeping a hundred
     * thresholds costs no more model evaluations than sweeping one.
     */
    fun sweep(
        outcomes: List<MultiLabelOutcome>,
        thresholds: List<Double> = DEFAULT_THRESHOLDS,
    ): List<ThresholdPoint> {
        require(value = outcomes.isNotEmpty()) { "cannot sweep an empty outcome list" }
        require(value = thresholds.isNotEmpty()) { "cannot sweep an empty threshold list" }
        return thresholds.sorted().map { threshold ->
            val shifted = outcomes.map { outcome ->
                MultiLabelOutcome(
                    expected = outcome.expected,
                    prediction = outcome.prediction.at(threshold = threshold),
                )
            }
            val metrics = MultiLabelMetricsFactory.from(outcomes = shifted)
            ThresholdPoint(
                threshold = threshold,
                precision = metrics.micro.precision,
                recall = metrics.micro.recall,
                f1 = metrics.micro.f1,
                coverage = metrics.coverage,
            )
        }
    }

    /**
     * Recall at each of [depths], ignoring the threshold entirely — the number that says whether a
     * reviewer shown *k* candidates would find the right label.
     */
    fun recallAtK(
        outcomes: List<MultiLabelOutcome>,
        depths: List<Int> = DEFAULT_RANK_DEPTHS,
    ): Map<Int, Double> {
        require(value = outcomes.isNotEmpty()) { "cannot measure recall over an empty outcome list" }
        return depths.sorted().associateWith { depth ->
            MultiLabelMetricsFactory.recallAtK(outcomes = outcomes, count = depth)
        }
    }
}
