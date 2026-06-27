package io.skein.classify.infrastructure

import java.lang.Long.rotateLeft

/**
 * SipHash-2-4: a fast keyed pseudo-random function. Used for feature hashing because it is
 * keyed (irreversible in aggregate, hash-flooding resistant) yet an order of magnitude faster
 * than HMAC-SHA-256 on the per-n-gram hot path.
 *
 * Reference: Aumasson & Bernstein, "SipHash: a fast short-input PRF" (2012).
 */
// MagicNumber suppressed: every literal here (initialization words, rotation amounts, masks) is
// a fixed constant mandated by the SipHash-2-4 specification; naming them adds noise, not clarity.
@Suppress("MagicNumber")
object SipHash {

    private const val COMPRESSION_ROUNDS = 2
    private const val FINALIZATION_ROUNDS = 4

    /** Computes the 64-bit SipHash-2-4 of [data] under the 128-bit key ([key0], [key1]). */
    fun hash(data: ByteArray, key0: Long, key1: Long): Long {
        var v0 = key0 xor 0x736f6d6570736575L
        var v1 = key1 xor 0x646f72616e646f6dL
        var v2 = key0 xor 0x6c7967656e657261L
        var v3 = key1 xor 0x7465646279746573L

        val end = data.size - (data.size % Long.SIZE_BYTES)
        var offset = 0
        while (offset < end) {
            val block = littleEndianLong(data = data, offset = offset)
            v3 = v3 xor block
            // SIPROUND, inlined (no per-round allocation, no lambda capture). v0..v3 mutate in place.
            for (round in 0 until COMPRESSION_ROUNDS) {
                v0 += v1; v1 = rotateLeft(v1, 13); v1 = v1 xor v0; v0 = rotateLeft(v0, 32)
                v2 += v3; v3 = rotateLeft(v3, 16); v3 = v3 xor v2
                v0 += v3; v3 = rotateLeft(v3, 21); v3 = v3 xor v0
                v2 += v1; v1 = rotateLeft(v1, 17); v1 = v1 xor v2; v2 = rotateLeft(v2, 32)
            }
            v0 = v0 xor block
            offset += Long.SIZE_BYTES
        }

        var lastBlock = (data.size.toLong() and 0xff) shl 56
        var shift = 0
        while (offset < data.size) {
            lastBlock = lastBlock or ((data[offset].toLong() and 0xff) shl shift)
            shift += 8
            offset++
        }
        v3 = v3 xor lastBlock
        for (round in 0 until COMPRESSION_ROUNDS) {
            v0 += v1; v1 = rotateLeft(v1, 13); v1 = v1 xor v0; v0 = rotateLeft(v0, 32)
            v2 += v3; v3 = rotateLeft(v3, 16); v3 = v3 xor v2
            v0 += v3; v3 = rotateLeft(v3, 21); v3 = v3 xor v0
            v2 += v1; v1 = rotateLeft(v1, 17); v1 = v1 xor v2; v2 = rotateLeft(v2, 32)
        }
        v0 = v0 xor lastBlock

        v2 = v2 xor 0xff
        for (round in 0 until FINALIZATION_ROUNDS) {
            v0 += v1; v1 = rotateLeft(v1, 13); v1 = v1 xor v0; v0 = rotateLeft(v0, 32)
            v2 += v3; v3 = rotateLeft(v3, 16); v3 = v3 xor v2
            v0 += v3; v3 = rotateLeft(v3, 21); v3 = v3 xor v0
            v2 += v1; v1 = rotateLeft(v1, 17); v1 = v1 xor v2; v2 = rotateLeft(v2, 32)
        }
        return v0 xor v1 xor v2 xor v3
    }

    private fun littleEndianLong(data: ByteArray, offset: Int): Long {
        var result = 0L
        for (index in 0 until Long.SIZE_BYTES) {
            result = result or ((data[offset + index].toLong() and 0xff) shl (8 * index))
        }
        return result
    }
}
