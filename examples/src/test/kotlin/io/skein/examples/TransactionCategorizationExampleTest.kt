package io.skein.examples

import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class TransactionCategorizationExampleTest {

    private val example = TransactionCategorizationExample()

    @Test
    internal fun `classifies an insurance transaction and extracts insurer-amount pairs`() {
        val result = example.process(purpose = "AIG-Life 67.89 Geico-Auto 120.00 insurance premium")
        assertEquals(expected = Label(value = "insurance"), actual = result.category)
        assertEquals(expected = listOf("AIG-Life", "Geico-Auto"), actual = result.extracted.valuesOf(name = "insurer"))
        assertEquals(expected = listOf("67.89", "120.00"), actual = result.extracted.valuesOf(name = "amount"))
    }

    @Test
    internal fun `classifies a rent transaction and extracts the customer number`() {
        val result = example.process(purpose = "rent apartment CustomerNumber CD456 monthly")
        assertEquals(expected = Label(value = "rent"), actual = result.category)
        assertEquals(expected = "CD456", actual = result.extracted.first(name = "customer")?.value)
    }

    @Test
    internal fun `classifies a salary transaction and routes no extraction`() {
        val result = example.process(purpose = "salary november payout")
        assertEquals(expected = Label(value = "salary"), actual = result.category)
        assertTrue(actual = result.extracted.fields.isEmpty(), message = "no slots routed for salary")
    }

    @Test
    internal fun `reports a confidence between zero and one`() {
        val result = example.process(purpose = "insurance premium policy")
        assertTrue(actual = result.confidence in 0.0..1.0)
    }
}
