package io.skein.classify.application

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class TokenMaskerTest {

    private val masker = TokenMasker()

    @Test
    internal fun `masks a date to its type`() {
        assertEquals(expected = "paid on <DATE> thanks", actual = masker.mask(text = "paid on 31.12.2024 thanks"))
    }

    @Test
    internal fun `masks an amount to its type`() {
        assertEquals(expected = "total <AMOUNT> eur", actual = masker.mask(text = "total 1.234,56 eur"))
    }

    @Test
    internal fun `masks a long reference number but keeps a short quantity`() {
        assertEquals(
            expected = "order <NUMERIC> qty 12",
            actual = masker.mask(text = "order 998877665 qty 12"),
        )
    }

    @Test
    internal fun `masks a long identifier but keeps a short code`() {
        assertEquals(
            expected = "ref <ALPHANUMERIC> box A1",
            actual = masker.mask(text = "ref AB12XY99 box A1"),
        )
    }

    @Test
    internal fun `keeps ordinary words untouched`() {
        assertEquals(
            expected = "monthly rent payment for apartment",
            actual = masker.mask(text = "monthly rent payment for apartment"),
        )
    }

    /**
     * The point of masking to a *type* rather than deleting: the record's shape still separates it
     * from one with no reference at all, which is real signal.
     */
    @Test
    internal fun `collapses distinct references to the same text while keeping them distinct from none`() {
        val first = masker.mask(text = "invoice 100200300 from acme")
        val second = masker.mask(text = "invoice 900800700 from acme")
        val none = masker.mask(text = "invoice from acme")

        assertEquals(expected = first, actual = second)
        assertTrue(actual = first != none)
    }

    @Test
    internal fun `respects a caller's length thresholds`() {
        val strict = TokenMasker(minNumericLength = 2)

        assertEquals(expected = "qty <NUMERIC>", actual = strict.mask(text = "qty 12"))
        assertEquals(expected = "qty 1", actual = strict.mask(text = "qty 1"))
    }

    @Test
    internal fun `handles empty and whitespace-only text`() {
        assertEquals(expected = "", actual = masker.mask(text = ""))
        assertEquals(expected = "", actual = masker.mask(text = "   "))
    }

    @Test
    internal fun `rejects a non-positive length threshold`() {
        assertFailsWith<IllegalArgumentException> { TokenMasker(minNumericLength = 0) }
        assertFailsWith<IllegalArgumentException> { TokenMasker(minAlphanumericLength = 0) }
    }
}
