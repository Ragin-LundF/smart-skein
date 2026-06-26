package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SchemaTest {

    @Test
    fun `builds a schema via the define DSL`() {
        val schema = Schema.define {
            text("purpose")
            categorical("counterparty")
            identifier("iban")
            label("category")
        }
        assertEquals(expected = "category", actual = schema.labelField.name)
        assertTrue(schema.field("iban") is IdentifierField)
        assertEquals(expected = SensitivityEnum.PII, actual = schema.field("iban")?.sensitivity)
    }

    @Test
    fun `rejects a schema without a label field`() {
        assertFailsWith<IllegalArgumentException> {
            Schema.define {
                text("purpose")
            }
        }
    }

    @Test
    fun `rejects a schema with more than one label field`() {
        assertFailsWith<IllegalArgumentException> {
            Schema.define {
                label("a")
                label("b")
            }
        }
    }
}
