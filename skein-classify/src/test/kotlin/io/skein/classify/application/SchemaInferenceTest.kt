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

internal class SchemaInferenceTest {

    private val inference = SchemaInference()

    private fun records(): List<Record> {
        return listOf(
            Record(
                values = mapOf(
                    "amount" to "12.50",
                    "city" to "berlin",
                    "note" to "monthly rent transfer",
                    "cat" to "housing",
                ),
            ),
            Record(
                values = mapOf(
                    "amount" to "980",
                    "city" to "munich",
                    "note" to "salary october payout",
                    "cat" to "income",
                ),
            ),
            Record(
                values = mapOf(
                    "amount" to "5",
                    "city" to "berlin",
                    "note" to "coffee at the corner shop",
                    "cat" to "food",
                ),
            ),
        )
    }

    @Test
    internal fun `infers numeric categorical text and label fields`() {
        val schema = inference.infer(records = records(), labelField = "cat")
        assertTrue(actual = schema.field(name = "amount") is NumericField)
        assertTrue(
            actual = schema.field(name = "city") is CategoricalField,
            message = "low-cardinality field is categorical",
        )
        assertTrue(actual = schema.field(name = "note") is TextField, message = "high-cardinality free text is text")
        assertTrue(actual = schema.field(name = "cat") is LabelField)
    }

    @Test
    internal fun `adds the label field even when absent from the sample`() {
        val schema = inference.infer(
            records = listOf(Record(values = mapOf("amount" to "1"))),
            labelField = "category",
        )
        assertTrue(actual = schema.field(name = "category") is LabelField)
    }

    @Test
    internal fun `rejects inference from zero records`() {
        assertFailsWith<IllegalArgumentException> {
            inference.infer(records = emptyList(), labelField = "cat")
        }
    }

    @Test
    internal fun `infers a PII identifier field from unique alphanumeric codes`() {
        val schema = inference.infer(
            records = listOf(
                Record(values = mapOf("customer" to "AB12345", "cat" to "x")),
                Record(values = mapOf("customer" to "CD67890", "cat" to "y")),
                Record(values = mapOf("customer" to "EF24680", "cat" to "z")),
            ),
            labelField = "cat",
        )
        assertTrue(actual = schema.field(name = "customer") is IdentifierField)
        assertEquals(expected = SensitivityEnum.PII, actual = schema.field(name = "customer")?.sensitivity)
    }
}
