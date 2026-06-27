package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

internal class RecordMapperTest {

    private val schema = Schema.define {
        text(name = "purpose")
        numeric(name = "amount")
        identifier(name = "iban")
        label(name = "category")
    }
    private val mapper = RecordMapper(schema = schema)

    @Test
    internal fun `concatenates feature-eligible fields and extracts the label`() {
        val mapped = mapper.map(
            record = Record(
                values = mapOf("purpose" to "Rent", "amount" to "1200", "iban" to "DE123", "category" to "housing"),
            ),
        )
        assertEquals(expected = "Rent 1200", actual = mapped.featureText)
        assertEquals(expected = Label(value = "housing"), actual = mapped.label)
    }

    @Test
    internal fun `excludes PII identifier fields from feature text`() {
        val mapped = mapper.map(
            record = Record(values = mapOf("purpose" to "rent", "iban" to "DE99999999", "category" to "housing")),
        )
        assertTrue(actual = "DE99999999" !in mapped.featureText, message = "identifier (PII) must not enter features")
    }

    @Test
    internal fun `returns a null label when the label value is blank`() {
        val mapped = mapper.map(record = Record(values = mapOf("purpose" to "rent", "category" to "  ")))
        assertNull(actual = mapped.label)
    }
}
