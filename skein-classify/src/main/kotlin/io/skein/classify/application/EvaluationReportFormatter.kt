package io.skein.classify.application

import io.skein.classify.domain.CrossValidationReport
import io.skein.classify.domain.EvaluationReport
import java.util.Locale

/** Off-diagonal confusion cells shown in the text report. */
private const val TOP_CONFUSION_LIMIT = 10

private const val LABEL_WIDTH = 28
private const val NUMBER_WIDTH = 10
private const val AVERAGE_NAME_WIDTH = 17

/**
 * Renders an [EvaluationReport] or [CrossValidationReport] as plain text.
 *
 * Lives in the library rather than in `skein-cli` because more than one consumer needs it and they
 * cannot share code any other way. It performs no I/O — it returns a [String] — and it formats with
 * [Locale.ROOT] so a report is byte-identical regardless of the default locale.
 *
 * ponytail: shows only the largest [TOP_CONFUSION_LIMIT] off-diagonal confusion cells, because a
 * full matrix over a few dozen labels is thousands of unreadable cells. Upgrade path: write the
 * whole matrix to CSV, which `skein-cli evaluate --confusion` does.
 */
object EvaluationReportFormatter {

    fun toText(report: EvaluationReport, title: String): String {
        val lines = ArrayList<String>()
        lines.add(element = title)
        lines.addAll(elements = headlineLines(report = report))
        lines.add(element = "")
        lines.addAll(elements = averagingLines(report = report))
        lines.add(element = "")
        lines.addAll(elements = perLabelLines(report = report))
        lines.add(element = "")
        lines.addAll(elements = calibrationLines(report = report))
        lines.add(element = "")
        lines.addAll(elements = confusionLines(report = report))
        return lines.joinToString(separator = "\n")
    }

    fun toText(report: CrossValidationReport, title: String): String {
        val accuracies = report.folds.joinToString(separator = " ") { fold -> number(value = fold.accuracy) }
        val summary = "  per-fold accuracy  $accuracies  (mean ${number(value = report.meanAccuracy())}, " +
            "sd ${number(value = report.accuracyStandardDeviation())})"
        return toText(report = report.pooled, title = title) + "\n\n" + summary
    }

    private fun headlineLines(report: EvaluationReport): List<String> {
        return listOf(
            "  samples        ${report.sampleCount}",
            "  labels         ${report.confusionMatrix.labels.size}",
            "  accuracy       ${number(value = report.accuracy)}" +
                "      (top-${report.topK}  ${number(value = report.topKAccuracy)})",
            "  log loss       ${number(value = report.logLoss)}",
            "  brier          ${number(value = report.brierScore)}",
            "  ECE            ${number(value = report.expectedCalibrationError)}",
        )
    }

    private fun averagingLines(report: EvaluationReport): List<String> {
        return listOf(
            "  " + "averaging".padEnd(length = AVERAGE_NAME_WIDTH) +
                pad(text = "precision") + pad(text = "recall") + pad(text = "f1"),
            averageRow(name = "macro", precision = report.macroAverage.precision,
                recall = report.macroAverage.recall, f1 = report.macroAverage.f1),
            averageRow(name = "micro", precision = report.microAverage.precision,
                recall = report.microAverage.recall, f1 = report.microAverage.f1),
            averageRow(name = "weighted", precision = report.weightedAverage.precision,
                recall = report.weightedAverage.recall, f1 = report.weightedAverage.f1),
        )
    }

    private fun averageRow(name: String, precision: Double, recall: Double, f1: Double): String {
        return "  " + name.padEnd(length = AVERAGE_NAME_WIDTH) +
            pad(text = number(value = precision)) + pad(text = number(value = recall)) + pad(text = number(value = f1))
    }

    private fun perLabelLines(report: EvaluationReport): List<String> {
        val header = "  " + "label".padEnd(length = LABEL_WIDTH) +
            pad(text = "precision") + pad(text = "recall") + pad(text = "f1") + pad(text = "support")
        val rows = report.perLabel.map { metrics ->
            "  " + metrics.label.value.padEnd(length = LABEL_WIDTH) +
                pad(text = number(value = metrics.precision)) +
                pad(text = number(value = metrics.recall)) +
                pad(text = number(value = metrics.f1)) +
                pad(text = metrics.support.toString())
        }
        return listOf(header) + rows
    }

    private fun calibrationLines(report: EvaluationReport): List<String> {
        val header = "  " + "calibration".padEnd(length = LABEL_WIDTH) +
            pad(text = "n") + pad(text = "mean conf") + pad(text = "accuracy")
        val rows = report.calibration.map { bin ->
            val range = "[${number(value = bin.lowerBound)},${number(value = bin.upperBound)})"
            "  " + range.padEnd(length = LABEL_WIDTH) +
                pad(text = bin.count.toString()) +
                pad(text = number(value = bin.meanConfidence)) +
                pad(text = number(value = bin.accuracy))
        }
        return listOf(header) + rows
    }

    private fun confusionLines(report: EvaluationReport): List<String> {
        val header = "  top confusions (expected -> predicted)"
        val rows = report.confusionMatrix.topConfusions(limit = TOP_CONFUSION_LIMIT).map { confusion ->
            val pair = "${confusion.first.value} -> ${confusion.second.value}"
            "  " + pair.padEnd(length = LABEL_WIDTH * 2) + confusion.third.toString()
        }
        return listOf(header) + rows.ifEmpty { listOf("  (none)") }
    }

    private fun pad(text: String): String {
        return text.padStart(length = NUMBER_WIDTH + 1)
    }

    private fun number(value: Double): String {
        return String.format(Locale.ROOT, "%.4f", value)
    }
}
