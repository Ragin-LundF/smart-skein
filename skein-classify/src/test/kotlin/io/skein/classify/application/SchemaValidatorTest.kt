package io.skein.classify.application

import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SchemaValidatorTest {

    private val schema = Schema.define {
        text("purpose")
        numeric("amount")
        label("category")
    }
    private val validator = SchemaValidator(schema)

    @Test
    fun `accepts a complete valid record`() {
        val result = validator.validate(
            Record(mapOf("purpose" to "rent", "amount" to "1200", "category" to "housing")),
        )
        assertTrue(result.isValid())
        assertTrue(result.warnings.isEmpty())
    }

    @Test
    fun `rejects a record with a missing label`() {
        val result = validator.validate(Record(mapOf("purpose" to "rent", "amount" to "1200")))
        assertFalse(result.isValid())
        assertEquals(expected = 1, actual = result.errors.size)
    }

    @Test
    fun `rejects a non-numeric value in a numeric field`() {
        val result = validator.validate(
            Record(mapOf("amount" to "lots", "category" to "housing")),
        )
        assertFalse(result.isValid())
        assertTrue(result.errors.any { error -> error.contains("amount") })
    }

    @Test
    fun `warns about missing and unknown fields without rejecting`() {
        val result = validator.validate(
            Record(mapOf("amount" to "10", "category" to "x", "extra" to "y")),
        )
        assertTrue(result.isValid())
        assertTrue(result.warnings.any { warning -> warning.contains("purpose") })
        assertTrue(result.warnings.any { warning -> warning.contains("extra") })
    }
}
