package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.spi.RecordSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class RecordImportServiceTest {

    private val schema = Schema.define {
        text(name = "purpose")
        numeric(name = "amount")
        label(name = "category")
    }

    private fun sourceOf(records: List<Record>): RecordSource {
        return object : RecordSource {
            override fun stream(): Sequence<Record> {
                return records.asSequence()
            }
        }
    }

    @Test
    internal fun `accepts valid records and rejects invalid ones`() {
        val service = RecordImportService(schema = schema)
        val result = service.importFrom(
            source = sourceOf(
                records = listOf(
                    Record(values = mapOf("purpose" to "rent", "amount" to "1200", "category" to "housing")),
                    Record(values = mapOf("purpose" to "missing label", "amount" to "5")),
                    Record(values = mapOf("purpose" to "bad amount", "amount" to "lots", "category" to "x")),
                ),
            ),
        )
        assertEquals(expected = 1, actual = result.acceptedCount())
        assertEquals(expected = 2, actual = result.rejectedCount())
        assertEquals(expected = Label(value = "housing"), actual = result.accepted.single().label)
    }

    @Test
    internal fun `collects warnings from accepted records`() {
        val service = RecordImportService(schema = schema)
        val result = service.importFrom(
            source = sourceOf(
                records = listOf(Record(values = mapOf("amount" to "10", "category" to "x", "extra" to "y"))),
            ),
        )
        assertEquals(expected = 1, actual = result.acceptedCount())
        assertTrue(actual = result.warnings.any { warning -> warning.contains(other = "extra") })
    }
}
