package io.skein.classify.application

import io.skein.classify.domain.CrossValidationReport
import io.skein.classify.domain.EvaluationReportFactory
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionOutcome
import io.skein.classify.domain.ScoredLabel
import kotlin.test.Test
import kotlin.test.assertTrue

internal class EvaluationReportFormatterTest {

    private fun outcome(expected: String, vararg scores: Pair<String, Double>): PredictionOutcome {
        val alternatives = scores
            .map { (label, probability) -> ScoredLabel(label = Label(value = label), probability = probability) }
            .sortedByDescending { scored -> scored.probability }
        return PredictionOutcome(
            expected = Label(value = expected),
            prediction = Prediction(
                label = alternatives.first().label,
                confidence = alternatives.first().probability,
                alternatives = alternatives,
            ),
        )
    }

    private fun sampleOutcomes(): List<PredictionOutcome> {
        return listOf(
            outcome("RENT", "RENT" to 0.9, "FOOD" to 0.1),
            outcome("RENT", "FOOD" to 0.6, "RENT" to 0.4),
            outcome("FOOD", "FOOD" to 0.8, "RENT" to 0.2),
        )
    }

    @Test
    internal fun `renders headline, averages, labels, calibration and confusions`() {
        val report = EvaluationReportFactory.from(outcomes = sampleOutcomes())
        val text = EvaluationReportFormatter.toText(report = report, title = "Evaluation — unit test")

        assertTrue(actual = text.startsWith(prefix = "Evaluation — unit test"))
        assertTrue(actual = text.contains(other = "samples        3"))
        assertTrue(actual = text.contains(other = "accuracy"))
        assertTrue(actual = text.contains(other = "macro"))
        assertTrue(actual = text.contains(other = "micro"))
        assertTrue(actual = text.contains(other = "weighted"))
        assertTrue(actual = text.contains(other = "RENT"))
        assertTrue(actual = text.contains(other = "FOOD"))
        assertTrue(actual = text.contains(other = "calibration"))
        assertTrue(actual = text.contains(other = "RENT -> FOOD"))
    }

    @Test
    internal fun `renders one row per calibration bin`() {
        val report = EvaluationReportFactory.from(outcomes = sampleOutcomes(), calibrationBins = 4)
        val text = EvaluationReportFormatter.toText(report = report, title = "t")

        assertTrue(actual = text.contains(other = "[0.0000,0.2500)"))
        assertTrue(actual = text.contains(other = "[0.7500,1.0000)"))
    }

    @Test
    internal fun `reports no confusions for a perfect model`() {
        val report = EvaluationReportFactory.from(
            outcomes = listOf(outcome("A", "A" to 1.0)),
        )
        val text = EvaluationReportFormatter.toText(report = report, title = "t")

        assertTrue(actual = text.contains(other = "(none)"))
    }

    @Test
    internal fun `cross-validation output adds the per-fold line`() {
        val pooled = EvaluationReportFactory.from(outcomes = sampleOutcomes())
        val report = CrossValidationReport(folds = listOf(pooled, pooled), pooled = pooled)
        val text = EvaluationReportFormatter.toText(report = report, title = "CV")

        assertTrue(actual = text.contains(other = "per-fold accuracy"))
        assertTrue(actual = text.contains(other = "mean"))
        assertTrue(actual = text.contains(other = "sd"))
    }
}
