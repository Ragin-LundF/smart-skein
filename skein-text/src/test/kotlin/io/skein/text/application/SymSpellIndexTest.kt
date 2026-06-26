package io.skein.text.application

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class SymSpellIndexTest {

    private val index = SymSpellIndex(words = setOf("apartment", "payment", "agreement"), maxEditDistance = 1)

    @Test
    internal fun `matches an exact word`() {
        assertEquals(expected = setOf("apartment"), actual = index.candidates(query = "apartment"))
    }

    @Test
    internal fun `matches a word missing one character`() {
        assertEquals(expected = setOf("apartment"), actual = index.candidates(query = "apartmnt"))
    }

    @Test
    internal fun `matches a single substitution`() {
        assertEquals(expected = setOf("payment"), actual = index.candidates(query = "paymant"))
    }

    @Test
    internal fun `returns nothing beyond the edit distance`() {
        assertTrue(actual = index.candidates(query = "xyzzy").isEmpty())
    }

    @Test
    internal fun `does not match across the distance limit`() {
        // "aprmnt" is two deletions from "apartment" — outside maxEditDistance = 1.
        assertTrue(actual = index.candidates(query = "aprmnt").isEmpty())
    }
}
