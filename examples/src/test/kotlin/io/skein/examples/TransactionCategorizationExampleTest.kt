package io.skein.examples

import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TransactionCategorizationExampleTest {

    private val example = TransactionCategorizationExample()

    @Test
    fun `classifies an insurance transaction and extracts insurer-amount pairs`() {
        val result = example.process("AIG-Life 67.89 Geico-Auto 120.00 insurance premium")
        assertEquals(expected = Label("insurance"), actual = result.category)
        assertEquals(expected = listOf("AIG-Life", "Geico-Auto"), actual = result.extracted.valuesOf("insurer"))
        assertEquals(expected = listOf("67.89", "120.00"), actual = result.extracted.valuesOf("amount"))
    }

    @Test
    fun `classifies a rent transaction and extracts the customer number`() {
        val result = example.process("rent apartment CustomerNumber CD456 monthly")
        assertEquals(expected = Label("rent"), actual = result.category)
        assertEquals(expected = "CD456", actual = result.extracted.first("customer")?.value)
    }

    @Test
    fun `classifies a salary transaction and routes no extraction`() {
        val result = example.process("salary november payout")
        assertEquals(expected = Label("salary"), actual = result.category)
        assertTrue(result.extracted.fields.isEmpty(), message = "no slots routed for salary")
    }

    @Test
    fun `reports a confidence between zero and one`() {
        val result = example.process("insurance premium policy")
        assertTrue(result.confidence in 0.0..1.0)
    }
}
