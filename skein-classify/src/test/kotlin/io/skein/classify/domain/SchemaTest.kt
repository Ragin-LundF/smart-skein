package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

internal class SchemaTest {

    @Test
    internal fun `builds a schema via the define DSL`() {
        val schema = Schema.define {
            text(name = "purpose")
            categorical(name = "counterparty")
            identifier(name = "iban")
            label(name = "category")
        }
        assertEquals(expected = "category", actual = schema.labelField.name)
        assertTrue(actual = schema.field(name = "iban") is IdentifierField)
        assertEquals(expected = SensitivityEnum.PII, actual = schema.field(name = "iban")?.sensitivity)
    }

    @Test
    internal fun `rejects a schema without a label field`() {
        assertFailsWith<IllegalArgumentException> {
            Schema.define {
                text(name = "purpose")
            }
        }
    }

    @Test
    internal fun `rejects a schema with more than one label field`() {
        assertFailsWith<IllegalArgumentException> {
            Schema.define {
                label(name = "a")
                label(name = "b")
            }
        }
    }

    @Test
    internal fun `HashingConfig randomKey produces a valid config with non-zero distinct keys`() {
        val config = HashingConfig.randomKey()
        assertTrue(actual = config.numFeatures > 0)
        // Two calls are astronomically unlikely to produce identical 64-bit random keys
        val other = HashingConfig.randomKey()
        assertNotEquals(illegal = config.key0, actual = other.key0)
    }
}
