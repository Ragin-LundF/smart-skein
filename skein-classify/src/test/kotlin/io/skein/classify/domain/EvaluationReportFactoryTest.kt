package io.skein.classify.domain

import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The metric conventions are hand-computed in the test comments so the assertions pin the intended
 * arithmetic rather than whatever the implementation currently produces.
 */
internal class EvaluationReportFactoryTest {

    private val tolerance = 1e-9

    private fun predictionOf(vararg scores: Pair<String, Double>): Prediction {
        val alternatives = scores
            .map { (label, probability) -> ScoredLabel(label = Label(value = label), probability = probability) }
            .sortedByDescending { scored -> scored.probability }
        return Prediction(
            label = alternatives.first().label,
            confidence = alternatives.first().probability,
            alternatives = alternatives,
        )
    }

    private fun outcome(expected: String, vararg scores: Pair<String, Double>): PredictionOutcome {
        return PredictionOutcome(expected = Label(value = expected), prediction = predictionOf(scores = scores))
    }

    /**
     * A deliberately imbalanced 3-class case, used by several tests below.
     *
     * Truth A x4 -> predicted A,A,A,B ; truth B x2 -> predicted B,A ; truth C x1 -> predicted A.
     *
     * tp: A=3 B=1 C=0. Column totals: A=5 B=2 C=0, so fp: A=2 B=1 C=0.
     * Support: A=4 B=2 C=1, so fn: A=1 B=1 C=1. Total 7, correct 4.
     */
    private fun imbalancedOutcomes(): List<PredictionOutcome> {
        return listOf(
            outcome("A", "A" to 0.8, "B" to 0.15, "C" to 0.05),
            outcome("A", "A" to 0.7, "B" to 0.2, "C" to 0.1),
            outcome("A", "A" to 0.6, "B" to 0.3, "C" to 0.1),
            outcome("A", "B" to 0.55, "A" to 0.35, "C" to 0.1),
            outcome("B", "B" to 0.9, "A" to 0.05, "C" to 0.05),
            outcome("B", "A" to 0.5, "B" to 0.4, "C" to 0.1),
            outcome("C", "A" to 0.65, "B" to 0.25, "C" to 0.1),
        )
    }

    @Test
    internal fun `a perfect classifier scores one everywhere and loses nothing`() {
        val outcomes = listOf(
            outcome("A", "A" to 1.0, "B" to 0.0),
            outcome("B", "B" to 1.0, "A" to 0.0),
        )
        val report = EvaluationReportFactory.from(outcomes = outcomes)

        assertEquals(expected = 2, actual = report.sampleCount)
        assertEquals(expected = 1.0, actual = report.accuracy, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = report.logLoss, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = report.brierScore, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = report.expectedCalibrationError, absoluteTolerance = tolerance)
        report.perLabel.forEach { metrics ->
            assertEquals(expected = 1.0, actual = metrics.precision, absoluteTolerance = tolerance)
            assertEquals(expected = 1.0, actual = metrics.recall, absoluteTolerance = tolerance)
            assertEquals(expected = 1.0, actual = metrics.f1, absoluteTolerance = tolerance)
        }
    }

    @Test
    internal fun `per-label metrics match the hand-computed imbalanced case`() {
        val report = EvaluationReportFactory.from(outcomes = imbalancedOutcomes())

        assertEquals(expected = 4.0 / 7.0, actual = report.accuracy, absoluteTolerance = tolerance)

        val a = report.metricsFor(label = Label(value = "A"))!!
        assertEquals(expected = 3.0 / 5.0, actual = a.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 3.0 / 4.0, actual = a.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 2.0 / 3.0, actual = a.f1, absoluteTolerance = tolerance)
        assertEquals(expected = 4, actual = a.support)

        val b = report.metricsFor(label = Label(value = "B"))!!
        assertEquals(expected = 0.5, actual = b.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.5, actual = b.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 0.5, actual = b.f1, absoluteTolerance = tolerance)
        assertEquals(expected = 2, actual = b.support)

        val c = report.metricsFor(label = Label(value = "C"))!!
        assertEquals(expected = 0.0, actual = c.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = c.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 1, actual = c.support)
    }

    @Test
    internal fun `micro average equals accuracy for single-label multi-class`() {
        val report = EvaluationReportFactory.from(outcomes = imbalancedOutcomes())

        assertEquals(expected = report.accuracy, actual = report.microAverage.precision, absoluteTolerance = tolerance)
        assertEquals(expected = report.accuracy, actual = report.microAverage.recall, absoluteTolerance = tolerance)
        assertEquals(expected = report.accuracy, actual = report.microAverage.f1, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `macro and weighted averages match the hand-computed imbalanced case`() {
        val report = EvaluationReportFactory.from(outcomes = imbalancedOutcomes())

        // macro = unweighted mean over A, B, C
        assertEquals(expected = 1.1 / 3.0, actual = report.macroAverage.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 1.25 / 3.0, actual = report.macroAverage.recall, absoluteTolerance = tolerance)

        // weighted = support-weighted mean; supports are A=4 B=2 C=1 over 7
        assertEquals(expected = 3.4 / 7.0, actual = report.weightedAverage.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 4.0 / 7.0, actual = report.weightedAverage.recall, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `macro average is dragged down by a label that is predicted but never expected`() {
        val withGhost = listOf(
            outcome("A", "A" to 0.9, "B" to 0.1),
            outcome("A", "ghost" to 0.6, "A" to 0.4),
        )
        val report = EvaluationReportFactory.from(outcomes = withGhost)

        // The matrix spans expected-union-PREDICTED labels only, so it holds A and ghost; B is
        // merely a ranked alternative and never wins or is expected.
        // A: precision 1/1 = 1.0. ghost: precision 0/1 = 0.0 with support 0.
        // Macro therefore halves to 0.5, where scoring A alone would have given 1.0.
        assertTrue(actual = report.perLabel.any { metrics -> metrics.label == Label(value = "ghost") })
        assertEquals(expected = 2, actual = report.perLabel.size)
        assertEquals(expected = 0.5, actual = report.macroAverage.precision, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `a label with no predictions scores zero precision rather than NaN`() {
        val report = EvaluationReportFactory.from(
            outcomes = listOf(outcome("A", "B" to 0.9, "A" to 0.1)),
        )
        val a = report.metricsFor(label = Label(value = "A"))!!
        assertEquals(expected = 0.0, actual = a.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = a.f1, absoluteTolerance = tolerance)
        assertTrue(actual = !report.macroAverage.precision.isNaN())
    }

    @Test
    internal fun `top-k counts a truth that is ranked below the winner`() {
        val outcomes = listOf(outcome("C", "A" to 0.5, "B" to 0.3, "C" to 0.2))

        assertEquals(
            expected = 0.0,
            actual = EvaluationReportFactory.from(outcomes = outcomes, topK = 1).topKAccuracy,
            absoluteTolerance = tolerance,
        )
        assertEquals(
            expected = 1.0,
            actual = EvaluationReportFactory.from(outcomes = outcomes, topK = 3).topKAccuracy,
            absoluteTolerance = tolerance,
        )
    }

    @Test
    internal fun `top-k of one reproduces accuracy and a large k does not overflow`() {
        val outcomes = imbalancedOutcomes()
        val topOne = EvaluationReportFactory.from(outcomes = outcomes, topK = 1)
        assertEquals(expected = topOne.accuracy, actual = topOne.topKAccuracy, absoluteTolerance = tolerance)

        val huge = EvaluationReportFactory.from(outcomes = outcomes, topK = 99)
        assertEquals(expected = 1.0, actual = huge.topKAccuracy, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `log loss stays finite when the true label is absent from the distribution`() {
        val report = EvaluationReportFactory.from(
            outcomes = listOf(outcome("Z", "A" to 0.9, "B" to 0.1)),
        )
        assertTrue(actual = report.logLoss.isFinite())
        assertEquals(expected = -ln(x = 1e-15), actual = report.logLoss, absoluteTolerance = 1e-6)
    }

    @Test
    internal fun `brier score matches the hand-computed two-outcome case`() {
        // (0.7-1)^2 + (0.3-0)^2 = 0.18 ; (0.6-0)^2 + (0.4-1)^2 = 0.72 ; mean = 0.45
        val report = EvaluationReportFactory.from(
            outcomes = listOf(
                outcome("A", "A" to 0.7, "B" to 0.3),
                outcome("B", "A" to 0.6, "B" to 0.4),
            ),
        )
        assertEquals(expected = 0.45, actual = report.brierScore, absoluteTolerance = tolerance)
        assertTrue(actual = report.brierScore in 0.0..2.0)
    }

    @Test
    internal fun `brier score charges a full unit when the true label has no output unit`() {
        val report = EvaluationReportFactory.from(
            outcomes = listOf(outcome("Z", "A" to 1.0)),
        )
        // (1.0 - 0)^2 for A, plus 1.0 for the missing true class.
        assertEquals(expected = 2.0, actual = report.brierScore, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `calibration bins place confidences including exactly one in range`() {
        val report = EvaluationReportFactory.from(
            outcomes = listOf(
                outcome("A", "A" to 0.05, "B" to 0.95).let { built ->
                    // force a low-confidence winner by expecting the loser
                    PredictionOutcome(expected = Label(value = "B"), prediction = built.prediction)
                },
                outcome("A", "A" to 1.0),
            ),
            calibrationBins = 10,
        )
        assertEquals(expected = 10, actual = report.calibration.size)
        // confidence 0.95 -> bin 9 ; confidence 1.0 -> clamped into bin 9 as well
        assertEquals(expected = 2, actual = report.calibration[9].count)
        assertEquals(expected = 0, actual = report.calibration[0].count)
        assertEquals(expected = 0.0, actual = report.calibration[0].meanConfidence, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `expected calibration error matches the hand-computed two-bin case`() {
        // conf 0.85 correct -> bin 8 ; conf 0.95 wrong -> bin 9
        // ECE = 0.5*|1.0-0.85| + 0.5*|0.0-0.95| = 0.075 + 0.475 = 0.55
        val report = EvaluationReportFactory.from(
            outcomes = listOf(
                outcome("A", "A" to 0.85, "B" to 0.15),
                outcome("B", "A" to 0.95, "B" to 0.05),
            ),
            calibrationBins = 10,
        )
        assertEquals(expected = 0.55, actual = report.expectedCalibrationError, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `rejects empty outcomes and non-positive parameters`() {
        assertFailsWith<IllegalArgumentException> { EvaluationReportFactory.from(outcomes = emptyList()) }
        assertFailsWith<IllegalArgumentException> {
            EvaluationReportFactory.from(outcomes = imbalancedOutcomes(), topK = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            EvaluationReportFactory.from(outcomes = imbalancedOutcomes(), calibrationBins = 0)
        }
    }
}
