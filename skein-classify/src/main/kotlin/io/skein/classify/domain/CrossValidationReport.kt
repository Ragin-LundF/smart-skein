package io.skein.classify.domain

import kotlin.math.sqrt

/**
 * The result of a k-fold cross-validation run.
 *
 * [folds] holds one report per fold, which is what reveals *variance* — a high mean with a wide
 * spread is a different situation from a high mean with a narrow one. [pooled] is built from every
 * fold's outcomes concatenated: because each observation is held out exactly once, it is the honest
 * confusion matrix for the run as a whole.
 */
data class CrossValidationReport(
    val folds: List<EvaluationReport>,
    val pooled: EvaluationReport,
) {

    init {
        require(value = folds.isNotEmpty()) { "a cross-validation report needs at least one fold" }
    }

    /** Mean accuracy across [folds]. */
    fun meanAccuracy(): Double {
        return folds.sumOf { fold -> fold.accuracy } / folds.size
    }

    /** Population standard deviation of the per-fold accuracies; `0.0` for a single fold. */
    fun accuracyStandardDeviation(): Double {
        val mean = meanAccuracy()
        val variance = folds.sumOf { fold -> (fold.accuracy - mean) * (fold.accuracy - mean) } / folds.size
        return sqrt(x = variance)
    }
}
