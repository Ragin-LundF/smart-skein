package io.skein.classify.application

import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class SchemaValidatorTest {

    private val schema = Schema.define {
        text(name = "purpose")
        numeric(name = "amount")
        label(name = "category")
    }
    private val validator = SchemaValidator(schema = schema)

    @Test
    internal fun `accepts a complete valid record`() {
        val result = validator.validate(
            record = Record(values = mapOf("purpose" to "rent", "amount" to "1200", "category" to "housing")),
        )
        assertTrue(actual = result.isValid())
        assertTrue(actual = result.warnings.isEmpty())
    }

    @Test
    internal fun `rejects a record with a missing label`() {
        val result = validator.validate(record = Record(values = mapOf("purpose" to "rent", "amount" to "1200")))
        assertFalse(actual = result.isValid())
        assertEquals(expected = 1, actual = result.errors.size)
    }

    @Test
    internal fun `rejects a non-numeric value in a numeric field`() {
        val result = validator.validate(
            record = Record(values = mapOf("amount" to "lots", "category" to "housing")),
        )
        assertFalse(actual = result.isValid())
        assertTrue(actual = result.errors.any { error -> error.contains(other = "amount") })
    }

    @Test
    internal fun `warns about missing and unknown fields without rejecting`() {
        val result = validator.validate(
            record = Record(values = mapOf("amount" to "10", "category" to "x", "extra" to "y")),
        )
        assertTrue(actual = result.isValid())
        assertTrue(actual = result.warnings.any { warning -> warning.contains(other = "purpose") })
        assertTrue(actual = result.warnings.any { warning -> warning.contains(other = "extra") })
    }
}
