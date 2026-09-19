package io.skein.classify.domain

import kotlin.math.sqrt

/**
 * The result of a multi-label k-fold cross-validation run.
 *
 * [folds] holds one reading per fold, which is what reveals *variance*: a high mean with a wide
 * spread is a different situation from a high mean with a narrow one, and only the per-fold numbers
 * distinguish them. [pooled] is built from every fold's outcomes concatenated — because each
 * observation is held out exactly once, it is the honest reading for the run as a whole.
 *
 * [pooledOutcomes] keeps the underlying scores so a threshold sweep costs no further model
 * evaluations: the same cross-validated predictions can be re-read at any cut. That is the table a
 * threshold should be chosen from, and re-running cross-validation per candidate threshold instead
 * is the expensive way to get the same answer.
 */
data class MultiLabelCrossValidationReport(
    val folds: List<MultiLabelMetrics>,
    val pooled: MultiLabelMetrics,
    val pooledOutcomes: List<MultiLabelOutcome>,
) {

    init {
        require(value = folds.isNotEmpty()) { "a cross-validation report needs at least one fold" }
    }

    /** Mean micro-F1 across [folds]. */
    fun meanMicroF1(): Double {
        return folds.sumOf { fold -> fold.micro.f1 } / folds.size
    }

    /**
     * Mean macro-F1 across [folds] — the number that moves when rare labels are handled badly, and
     * therefore the one worth watching on a long-tailed taxonomy.
     */
    fun meanMacroF1(): Double {
        return folds.sumOf { fold -> fold.macro.f1 } / folds.size
    }

    /** Population standard deviation of the per-fold micro-F1; `0.0` for a single fold. */
    fun microF1StandardDeviation(): Double {
        val mean = meanMicroF1()
        val variance = folds.sumOf { fold -> (fold.micro.f1 - mean) * (fold.micro.f1 - mean) } / folds.size
        return sqrt(x = variance)
    }
}
