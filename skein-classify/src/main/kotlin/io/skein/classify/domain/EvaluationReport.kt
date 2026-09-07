package io.skein.classify.domain

/**
 * What a model scored on a set of evaluated cases.
 *
 * Conventions worth knowing before reading the numbers:
 * - [macroAverage] averages over **every** label in [confusionMatrix], including labels the model
 *   predicted but that were never the truth. A hallucinated label therefore drags it down. This is
 *   stricter than the common "average over labels with support" convention.
 * - [microAverage] aggregates true/false positives and negatives before dividing. For single-label
 *   multi-class problems its precision, recall and F1 all provably equal [accuracy].
 * - [weightedAverage] weights each label's metric by its support.
 * - [logLoss] clamps the true label's probability at `1e-15`, so a label the model has never seen
 *   yields a large but finite penalty instead of infinity.
 * - [brierScore] is the multi-class Brier score in `[0, 2]`; lower is better.
 * - [expectedCalibrationError] is the support-weighted mean gap between confidence and accuracy
 *   across [calibration]; `0.0` means perfectly calibrated.
 */
data class EvaluationReport(
    val sampleCount: Int,
    val accuracy: Double,
    val topK: Int,
    val topKAccuracy: Double,
    val logLoss: Double,
    val brierScore: Double,
    val expectedCalibrationError: Double,
    val perLabel: List<LabelMetrics>,
    val macroAverage: AveragedMetrics,
    val microAverage: AveragedMetrics,
    val weightedAverage: AveragedMetrics,
    val confusionMatrix: ConfusionMatrix,
    val calibration: List<CalibrationBin>,
) {

    /** Metrics for [label], or `null` when it neither appeared nor was predicted. */
    fun metricsFor(label: Label): LabelMetrics? {
        return perLabel.firstOrNull { metrics -> metrics.label == label }
    }
}
