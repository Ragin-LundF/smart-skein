package io.skein.text.application

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SymSpellIndexTest {

    private val index = SymSpellIndex(setOf("apartment", "payment", "agreement"), maxEditDistance = 1)

    @Test
    fun `matches an exact word`() {
        assertEquals(expected = setOf("apartment"), actual = index.candidates("apartment"))
    }

    @Test
    fun `matches a word missing one character`() {
        assertEquals(expected = setOf("apartment"), actual = index.candidates("apartmnt"))
    }

    @Test
    fun `matches a single substitution`() {
        assertEquals(expected = setOf("payment"), actual = index.candidates("paymant"))
    }

    @Test
    fun `returns nothing beyond the edit distance`() {
        assertTrue(index.candidates("xyzzy").isEmpty())
    }

    @Test
    fun `does not match across the distance limit`() {
        // "aprmnt" is two deletions from "apartment" — outside maxEditDistance = 1.
        assertTrue(index.candidates("aprmnt").isEmpty())
    }
}
