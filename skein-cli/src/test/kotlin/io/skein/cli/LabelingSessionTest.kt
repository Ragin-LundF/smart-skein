package io.skein.cli

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Schema
import io.skein.classify.domain.UncertaintyStrategyEnum
import java.io.BufferedReader
import java.io.StringReader
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LabelingSessionTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }
    private val hashing = HashingConfig(key0 = 7L, key1 = 13L)

    @Test
    fun `accepting suggestions labels every uncertain row and the engine learns them`() {
        val rows = sampleRows()
        val unlabeled = rows.count { row -> (row["category"] as String).isEmpty() }

        val labeled = session(input = "y\n".repeat(n = 10)).run(rows = rows)

        assertEquals(expected = unlabeled, actual = labeled)
        assertTrue(actual = rows.all { row -> (row["category"] as String).isNotBlank() }, message = "all rows labeled")
    }

    @Test
    fun `quit stops immediately and labels nothing`() {
        val rows = sampleRows()

        val labeled = session(input = "q\n").run(rows = rows)

        assertEquals(expected = 0, actual = labeled)
        assertTrue(actual = rows.any { row -> (row["category"] as String).isEmpty() }, message = "rows unlabeled")
    }

    @Test
    fun `skip leaves one row unlabeled while the rest are labeled`() {
        val rows = sampleRows()
        val unlabeled = rows.count { row -> (row["category"] as String).isEmpty() }

        val labeled = session(input = "s\n" + "y\n".repeat(n = 10)).run(rows = rows)

        assertEquals(expected = unlabeled - 1, actual = labeled)
        assertEquals(expected = 1, actual = rows.count { row -> (row["category"] as String).isEmpty() })
    }

    @Test
    fun `typed label overrides the suggestion`() {
        val rows = sampleRows()

        session(input = "groceries\n".repeat(n = 10)).run(rows = rows)

        val target = rows.filter { row -> row["purpose"] == "supermarket food" }
        assertTrue(actual = target.all { row -> row["category"] == "groceries" }, message = "typed label applied")
    }

    private fun session(input: String): LabelingSession {
        val engine = CliEngine.fresh(
            schema = schema,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = hashing,
        )
        return LabelingSession(
            engine = engine,
            budget = 100,
            batchSize = 8,
            strategy = UncertaintyStrategyEnum.MARGIN,
            epochs = 1,
            input = BufferedReader(StringReader(input)),
            output = StringBuilder(),
        )
    }

    private fun sampleRows(): List<MutableMap<String, Any?>> {
        return listOf(
            row(purpose = "insurance premium annual policy", category = "insurance"),
            row(purpose = "rent apartment payment", category = "rent"),
            row(purpose = "Allianz insurance premium", category = ""),
            row(purpose = "rent flat monthly", category = ""),
            row(purpose = "supermarket food", category = ""),
        )
    }

    private fun row(purpose: String, category: String): MutableMap<String, Any?> {
        return linkedMapOf("purpose" to purpose, "category" to category)
    }
}
