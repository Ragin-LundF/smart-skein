package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class CrossValidationReportTest {

    private val tolerance = 1e-9

    private fun reportWithAccuracy(correct: Int, wrong: Int): EvaluationReport {
        val outcomes = List(size = correct) {
            PredictionOutcome(
                expected = Label(value = "A"),
                prediction = Prediction(
                    label = Label(value = "A"),
                    confidence = 1.0,
                    alternatives = listOf(ScoredLabel(label = Label(value = "A"), probability = 1.0)),
                ),
            )
        } + List(size = wrong) {
            PredictionOutcome(
                expected = Label(value = "B"),
                prediction = Prediction(
                    label = Label(value = "A"),
                    confidence = 1.0,
                    alternatives = listOf(ScoredLabel(label = Label(value = "A"), probability = 1.0)),
                ),
            )
        }
        return EvaluationReportFactory.from(outcomes = outcomes)
    }

    @Test
    internal fun `mean accuracy averages the folds`() {
        val folds = listOf(
            reportWithAccuracy(correct = 8, wrong = 2),
            reportWithAccuracy(correct = 6, wrong = 4),
        )
        val report = CrossValidationReport(folds = folds, pooled = reportWithAccuracy(correct = 14, wrong = 6))

        assertEquals(expected = 0.7, actual = report.meanAccuracy(), absoluteTolerance = tolerance)
    }

    @Test
    internal fun `standard deviation is the population spread of fold accuracies`() {
        val folds = listOf(
            reportWithAccuracy(correct = 8, wrong = 2),
            reportWithAccuracy(correct = 6, wrong = 4),
        )
        val report = CrossValidationReport(folds = folds, pooled = reportWithAccuracy(correct = 14, wrong = 6))

        // accuracies 0.8 and 0.6, mean 0.7, population sd = 0.1
        assertEquals(expected = 0.1, actual = report.accuracyStandardDeviation(), absoluteTolerance = tolerance)
    }

    @Test
    internal fun `a single fold has zero spread`() {
        val fold = reportWithAccuracy(correct = 7, wrong = 3)
        val report = CrossValidationReport(folds = listOf(fold), pooled = fold)

        assertEquals(expected = 0.0, actual = report.accuracyStandardDeviation(), absoluteTolerance = tolerance)
    }

    @Test
    internal fun `rejects an empty fold list`() {
        assertFailsWith<IllegalArgumentException> {
            CrossValidationReport(folds = emptyList(), pooled = reportWithAccuracy(correct = 1, wrong = 0))
        }
    }
}
