package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
}
