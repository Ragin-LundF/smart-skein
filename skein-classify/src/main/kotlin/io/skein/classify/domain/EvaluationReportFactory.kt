package io.skein.classify.domain

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/** Default number of ranked alternatives considered by top-k accuracy. */
private const val DEFAULT_TOP_K = 3

/** Default number of equal-width buckets in the reliability diagram. */
private const val DEFAULT_CALIBRATION_BINS = 10

/**
 * Floor applied to the true label's probability before taking its logarithm. Without it, a single
 * outcome whose true label the model has never seen would make the whole log loss infinite.
 */
private const val LOG_LOSS_EPSILON = 1e-15

/**
 * Turns evaluated cases into an [EvaluationReport].
 *
 * Deliberately a pure function of [PredictionOutcome]s: it never touches a classifier, a feature
 * store or a file, so the metric conventions can be verified against hand-built predictions with
 * no model and no fixtures. Mirrors [PredictionFactory], the module's other piece of shared
 * scoring math.
 *
 * The conventions it implements are documented on [EvaluationReport].
 */
object EvaluationReportFactory {

    /**
     * Scores [outcomes]. [topK] is the rank depth for top-k accuracy (`1` reproduces plain
     * accuracy); [calibrationBins] is the number of equal-width reliability buckets.
     */
    fun from(
        outcomes: List<PredictionOutcome>,
        topK: Int = DEFAULT_TOP_K,
        calibrationBins: Int = DEFAULT_CALIBRATION_BINS,
    ): EvaluationReport {
        require(value = outcomes.isNotEmpty()) { "cannot evaluate an empty outcome list" }
        require(value = topK >= 1) { "topK must be at least 1" }
        require(value = calibrationBins >= 1) { "calibrationBins must be at least 1" }

        val matrix = ConfusionMatrix.of(outcomes = outcomes)
        val perLabel = labelMetrics(matrix = matrix)
        val bins = calibrationOf(outcomes = outcomes, binCount = calibrationBins)
        return EvaluationReport(
            sampleCount = outcomes.size,
            accuracy = accuracyOf(matrix = matrix),
            topK = topK,
            topKAccuracy = topKAccuracyOf(outcomes = outcomes, topK = topK),
            logLoss = logLossOf(outcomes = outcomes),
            brierScore = brierOf(outcomes = outcomes),
            expectedCalibrationError = eceOf(bins = bins, total = outcomes.size),
            perLabel = perLabel,
            macroAverage = macroOf(perLabel = perLabel),
            microAverage = microOf(matrix = matrix),
            weightedAverage = weightedOf(perLabel = perLabel),
            confusionMatrix = matrix,
            calibration = bins,
        )
    }

    private fun accuracyOf(matrix: ConfusionMatrix): Double {
        val correct = matrix.labels.sumOf { label -> matrix.correct(label = label) }
        return ratio(numerator = correct.toDouble(), denominator = matrix.total().toDouble())
    }

    private fun labelMetrics(matrix: ConfusionMatrix): List<LabelMetrics> {
        return matrix.labels.map { label ->
            val truePositives = matrix.correct(label = label)
            val falsePositives = matrix.columnTotal(label = label) - truePositives
            val falseNegatives = matrix.rowTotal(label = label) - truePositives
            val precision = ratio(
                numerator = truePositives.toDouble(),
                denominator = (truePositives + falsePositives).toDouble(),
            )
            val recall = ratio(
                numerator = truePositives.toDouble(),
                denominator = (truePositives + falseNegatives).toDouble(),
            )
            LabelMetrics(
                label = label,
                precision = precision,
                recall = recall,
                f1 = harmonicMean(precision = precision, recall = recall),
                support = matrix.rowTotal(label = label),
            )
        }
    }

    private fun macroOf(perLabel: List<LabelMetrics>): AveragedMetrics {
        val count = perLabel.size.toDouble()
        val precision = ratio(numerator = perLabel.sumOf { metrics -> metrics.precision }, denominator = count)
        val recall = ratio(numerator = perLabel.sumOf { metrics -> metrics.recall }, denominator = count)
        return AveragedMetrics(
            precision = precision,
            recall = recall,
            f1 = ratio(numerator = perLabel.sumOf { metrics -> metrics.f1 }, denominator = count),
        )
    }

    private fun microOf(matrix: ConfusionMatrix): AveragedMetrics {
        var truePositives = 0
        var falsePositives = 0
        var falseNegatives = 0
        for (label in matrix.labels) {
            val correct = matrix.correct(label = label)
            truePositives += correct
            falsePositives += matrix.columnTotal(label = label) - correct
            falseNegatives += matrix.rowTotal(label = label) - correct
        }
        val precision = ratio(
            numerator = truePositives.toDouble(),
            denominator = (truePositives + falsePositives).toDouble(),
        )
        val recall = ratio(
            numerator = truePositives.toDouble(),
            denominator = (truePositives + falseNegatives).toDouble(),
        )
        return AveragedMetrics(
            precision = precision,
            recall = recall,
            f1 = harmonicMean(precision = precision, recall = recall),
        )
    }

    private fun weightedOf(perLabel: List<LabelMetrics>): AveragedMetrics {
        val support = perLabel.sumOf { metrics -> metrics.support }.toDouble()
        val precision = perLabel.sumOf { metrics -> metrics.precision * metrics.support }
        val recall = perLabel.sumOf { metrics -> metrics.recall * metrics.support }
        val f1 = perLabel.sumOf { metrics -> metrics.f1 * metrics.support }
        return AveragedMetrics(
            precision = ratio(numerator = precision, denominator = support),
            recall = ratio(numerator = recall, denominator = support),
            f1 = ratio(numerator = f1, denominator = support),
        )
    }

    private fun topKAccuracyOf(outcomes: List<PredictionOutcome>, topK: Int): Double {
        val hits = outcomes.count { outcome ->
            outcome.prediction.alternatives
                .take(n = topK)
                .any { scored -> scored.label == outcome.expected }
        }
        return ratio(numerator = hits.toDouble(), denominator = outcomes.size.toDouble())
    }

    private fun logLossOf(outcomes: List<PredictionOutcome>): Double {
        val total = outcomes.sumOf { outcome ->
            ln(x = max(a = probabilityOfExpected(outcome = outcome), b = LOG_LOSS_EPSILON))
        }
        return -total / outcomes.size
    }

    private fun brierOf(outcomes: List<PredictionOutcome>): Double {
        val total = outcomes.sumOf { outcome ->
            var squaredError = 0.0
            var expectedSeen = false
            for (scored in outcome.prediction.alternatives) {
                val target = if (scored.label == outcome.expected) 1.0 else 0.0
                if (scored.label == outcome.expected) {
                    expectedSeen = true
                }
                squaredError += (scored.probability - target) * (scored.probability - target)
            }
            // The model has no output unit for the true label: full squared error on that class.
            if (expectedSeen) squaredError else squaredError + 1.0
        }
        return total / outcomes.size
    }

    private fun calibrationOf(outcomes: List<PredictionOutcome>, binCount: Int): List<CalibrationBin> {
        val width = 1.0 / binCount
        val counts = IntArray(size = binCount)
        val confidenceSums = DoubleArray(size = binCount)
        val correctCounts = IntArray(size = binCount)
        for (outcome in outcomes) {
            val confidence = outcome.prediction.confidence
            // The clamp is what places a confidence of exactly 1.0 in the last bin.
            val index = min(a = floor(x = confidence * binCount).toInt(), b = binCount - 1)
            counts[index]++
            confidenceSums[index] += confidence
            if (outcome.prediction.label == outcome.expected) {
                correctCounts[index]++
            }
        }
        return (0 until binCount).map { index ->
            val count = counts[index].toDouble()
            CalibrationBin(
                lowerBound = index * width,
                upperBound = (index + 1) * width,
                count = counts[index],
                meanConfidence = ratio(numerator = confidenceSums[index], denominator = count),
                accuracy = ratio(numerator = correctCounts[index].toDouble(), denominator = count),
            )
        }
    }

    private fun eceOf(bins: List<CalibrationBin>, total: Int): Double {
        return bins
            .filter { bin -> bin.count > 0 }
            .sumOf { bin -> bin.count.toDouble() / total * abs(x = bin.accuracy - bin.meanConfidence) }
    }

    private fun probabilityOfExpected(outcome: PredictionOutcome): Double {
        val scored = outcome.prediction.alternatives.firstOrNull { candidate -> candidate.label == outcome.expected }
        return scored?.probability ?: 0.0
    }

    private fun harmonicMean(precision: Double, recall: Double): Double {
        return ratio(numerator = 2.0 * precision * recall, denominator = precision + recall)
    }

    /** Division that yields `0.0` rather than `NaN` on a zero denominator. */
    private fun ratio(numerator: Double, denominator: Double): Double {
        return if (denominator == 0.0) 0.0 else numerator / denominator
    }
}
