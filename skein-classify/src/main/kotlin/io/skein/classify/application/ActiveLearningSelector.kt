package io.skein.classify.application

import io.skein.classify.domain.Prediction
import io.skein.classify.domain.Record
import io.skein.classify.domain.ReviewCandidate
import io.skein.classify.domain.UncertaintyStrategyEnum
import kotlin.math.ln

/**
 * Selects the records a trained engine is least certain about so a human can label them — the
 * core of active learning. The uncertainty measure is configurable via [UncertaintyStrategyEnum]
 * (margin sampling by default); the most uncertain candidates come first.
 */
class ActiveLearningSelector(private val service: ClassificationService) {

    /** Returns up to [limit] candidates ordered most-uncertain first, per [strategy]. */
    fun selectForReview(
        candidates: List<Record>,
        limit: Int,
        strategy: UncertaintyStrategyEnum = UncertaintyStrategyEnum.MARGIN,
    ): List<ReviewCandidate> {
        require(limit >= 0) { "limit must not be negative" }
        return candidates
            .map { record -> toCandidate(record) }
            .sortedByDescending { candidate -> uncertaintyOf(candidate.prediction, strategy) }
            .take(limit)
    }

    private fun toCandidate(record: Record): ReviewCandidate {
        val prediction = service.classify(record)
        return ReviewCandidate(record = record, prediction = prediction, margin = marginOf(prediction))
    }

    private fun uncertaintyOf(prediction: Prediction, strategy: UncertaintyStrategyEnum): Double {
        return when (strategy) {
            UncertaintyStrategyEnum.MARGIN -> -marginOf(prediction)
            UncertaintyStrategyEnum.LEAST_CONFIDENCE -> 1.0 - prediction.confidence
            UncertaintyStrategyEnum.ENTROPY -> entropyOf(prediction)
        }
    }

    private fun marginOf(prediction: Prediction): Double {
        val alternatives = prediction.alternatives
        if (alternatives.size < 2) {
            return 1.0
        }
        return alternatives[0].probability - alternatives[1].probability
    }

    private fun entropyOf(prediction: Prediction): Double {
        var entropy = 0.0
        for (scored in prediction.alternatives) {
            val probability = scored.probability
            if (probability > 0.0) {
                entropy -= probability * ln(probability)
            }
        }
        return entropy
    }
}
