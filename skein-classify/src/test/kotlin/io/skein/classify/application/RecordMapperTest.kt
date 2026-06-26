package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RecordMapperTest {

    private val schema = Schema.define {
        text("purpose")
        numeric("amount")
        identifier("iban")
        label("category")
    }
    private val mapper = RecordMapper(schema)

    @Test
    fun `concatenates feature-eligible fields and extracts the label`() {
        val mapped = mapper.map(
            Record(mapOf("purpose" to "Rent", "amount" to "1200", "iban" to "DE123", "category" to "housing")),
        )
        assertEquals(expected = "Rent 1200", actual = mapped.featureText)
        assertEquals(expected = Label("housing"), actual = mapped.label)
    }

    @Test
    fun `excludes PII identifier fields from feature text`() {
        val mapped = mapper.map(
            Record(mapOf("purpose" to "rent", "iban" to "DE99999999", "category" to "housing")),
        )
        assertTrue("DE99999999" !in mapped.featureText, message = "identifier (PII) must not enter features")
    }

    @Test
    fun `returns a null label when the label value is blank`() {
        val mapped = mapper.map(Record(mapOf("purpose" to "rent", "category" to "  ")))
        assertNull(mapped.label)
    }
}
