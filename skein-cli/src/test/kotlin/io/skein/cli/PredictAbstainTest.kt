package io.skein.cli

import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.domain.Calibration
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class PredictAbstainTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }

    private fun trainedEngine(): CliEngine {
        val engine = CliEngine.fresh(
            schema = schema,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = HashingConfig(key0 = 13L, key1 = 17L),
        )
        listOf(
            "rent transfer landlord" to "RENT",
            "monthly rent landlord" to "RENT",
            "supermarket groceries" to "FOOD",
            "groceries market food" to "FOOD",
        ).forEach { (purpose, category) ->
            engine.service.learn(record = Record(values = mapOf("purpose" to purpose, "category" to category)))
        }
        return engine
    }

    private fun rows(): List<MutableMap<String, Any?>> {
        return listOf(
            mutableMapOf<String, Any?>("purpose" to "rent transfer landlord", "category" to ""),
            mutableMapOf<String, Any?>("purpose" to "supermarket groceries", "category" to ""),
        )
    }

    @Test
    internal fun `a zero threshold labels every row`() {
        val rows = rows()
        val abstained = predictRows(
            engine = trainedEngine(),
            rows = rows,
            labelColumn = "category",
            confidenceColumn = "category_confidence",
            minConfidence = 0.0,
        )

        assertEquals(expected = 0, actual = abstained)
        rows.forEach { row -> assertTrue(actual = (row["category"] as String).isNotEmpty()) }
    }

    @Test
    internal fun `an unreachable threshold blanks the label but keeps the confidence`() {
        val engine = trainedEngine()
        // Soften the model so no row can clear a high bar.
        engine.service.calibration = Calibration(temperature = 500.0)
        val rows = rows()

        val abstained = predictRows(
            engine = engine,
            rows = rows,
            labelColumn = "category",
            confidenceColumn = "category_confidence",
            minConfidence = 0.99,
        )

        assertEquals(expected = rows.size, actual = abstained)
        rows.forEach { row ->
            assertEquals(expected = "", actual = row["category"])
            assertTrue(actual = (row["category_confidence"] as String).isNotEmpty())
        }
    }

    @Test
    internal fun `abstained rows come back as pending for the labeling session`() {
        val engine = trainedEngine()
        engine.service.calibration = Calibration(temperature = 500.0)
        val rows = rows()
        predictRows(
            engine = engine,
            rows = rows,
            labelColumn = "category",
            confidenceColumn = "category_confidence",
            minConfidence = 0.99,
        )

        // An empty label cell is what `label` treats as still needing a human.
        val pending = rows.count { row -> (row["category"] as? String).isNullOrBlank() }
        assertEquals(expected = rows.size, actual = pending)
    }
}
