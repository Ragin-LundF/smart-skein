package io.skein.classify.infrastructure

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

internal class SipHashTest {

    // Reference key from the SipHash paper: bytes 00 01 02 ... 0f as two little-endian 64-bit words.
    private val key0 = 0x0706050403020100L
    private val key1 = 0x0f0e0d0c0b0a0908L

    @Test
    internal fun `matches the reference vector for the empty message`() {
        // SipHash-2-4 test vector 0 from the reference implementation.
        assertEquals(
            expected = 0x726fdb47dd0e0e31L,
            actual = SipHash.hash(data = ByteArray(size = 0), key0 = key0, key1 = key1),
        )
    }

    @Test
    internal fun `matches the reference vector for a fifteen byte message`() {
        // Message 00 01 02 ... 0e (15 bytes) -> reference vector 15.
        val message = ByteArray(size = 15) { index -> index.toByte() }
        assertEquals(
            expected = 0xa129ca6149be45e5UL.toLong(),
            actual = SipHash.hash(data = message, key0 = key0, key1 = key1),
        )
    }

    @Test
    internal fun `different keys produce different hashes for the same input`() {
        val message = "skein".toByteArray()
        assertNotEquals(
            illegal = SipHash.hash(data = message, key0 = key0, key1 = key1),
            actual = SipHash.hash(data = message, key0 = 1L, key1 = 2L),
        )
    }
}
