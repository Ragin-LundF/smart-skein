package io.skein.classify.application

import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.Record
import io.skein.classify.domain.SensitivityEnum
import io.skein.classify.domain.TextField
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SchemaInferenceTest {

    private val inference = SchemaInference()

    private fun records(): List<Record> {
        return listOf(
            Record(
                mapOf("amount" to "12.50", "city" to "berlin", "note" to "monthly rent transfer", "cat" to "housing"),
            ),
            Record(
                mapOf("amount" to "980", "city" to "munich", "note" to "salary october payout", "cat" to "income"),
            ),
            Record(
                mapOf("amount" to "5", "city" to "berlin", "note" to "coffee at the corner shop", "cat" to "food"),
            ),
        )
    }

    @Test
    fun `infers numeric categorical text and label fields`() {
        val schema = inference.infer(records(), labelField = "cat")
        assertTrue(schema.field("amount") is NumericField)
        assertTrue(schema.field("city") is CategoricalField, message = "low-cardinality field is categorical")
        assertTrue(schema.field("note") is TextField, message = "high-cardinality free text is text")
        assertTrue(schema.field("cat") is LabelField)
    }

    @Test
    fun `adds the label field even when absent from the sample`() {
        val schema = inference.infer(
            listOf(Record(mapOf("amount" to "1"))),
            labelField = "category",
        )
        assertTrue(schema.field("category") is LabelField)
    }

    @Test
    fun `rejects inference from zero records`() {
        assertFailsWith<IllegalArgumentException> {
            inference.infer(emptyList(), labelField = "cat")
        }
    }

    @Test
    fun `infers a PII identifier field from unique alphanumeric codes`() {
        val schema = inference.infer(
            listOf(
                Record(mapOf("customer" to "AB12345", "cat" to "x")),
                Record(mapOf("customer" to "CD67890", "cat" to "y")),
                Record(mapOf("customer" to "EF24680", "cat" to "z")),
            ),
            labelField = "cat",
        )
        assertTrue(schema.field("customer") is IdentifierField)
        assertEquals(expected = SensitivityEnum.PII, actual = schema.field("customer")?.sensitivity)
    }
}
