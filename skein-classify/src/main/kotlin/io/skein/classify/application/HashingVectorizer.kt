package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.infrastructure.IntFloatHashMap
import io.skein.classify.infrastructure.SipHash
import io.skein.text.infrastructure.DefaultTextNormalizer
import io.skein.text.spi.TextNormalizer

/**
 * Turns text into a sparse [FeatureVector] of hashed character and word n-grams.
 *
 * The representation is **order-free** (a bag of n-grams) and **irreversible**: each n-gram is
 * keyed-hashed with SipHash into a fixed feature space, so the original content cannot be
 * recovered from the indices. Normalization makes the features typo- and spacing-tolerant.
 *
 * Counts accumulate into a reusable per-thread [IntFloatHashMap] (no boxing, no per-call map
 * allocation) and are emitted as sorted parallel arrays.
 */
class HashingVectorizer(
    private val config: HashingConfig,
    private val normalizer: TextNormalizer = DefaultTextNormalizer(),
) {

    private val scratch = ThreadLocal.withInitial { IntFloatHashMap() }

    // Reusable UTF-8 encode buffer per thread. Max char n-gram: charNgramMax chars × 4 UTF-8 bytes.
    // 256 bytes covers word n-grams up to ~40 chars without reallocation.
    private val encBuf = ThreadLocal.withInitial { ByteArray(256) }

    fun vectorize(text: String): FeatureVector {
        val normalized = normalizer.normalize(raw = text)
        val accumulator = scratch.get()
        val buf = encBuf.get()
        accumulator.clear()
        addCharNgrams(text = normalized, accumulator = accumulator, buf = buf)
        addWordNgrams(text = normalized, accumulator = accumulator, buf = buf)
        val (indices, values) = accumulator.sortedKeysAndValues()
        return FeatureVector(indices = indices, values = values)
    }

    private fun addCharNgrams(text: String, accumulator: IntFloatHashMap, buf: ByteArray) {
        if (text.isEmpty()) {
            return
        }
        for (size in config.charNgramMin..config.charNgramMax) {
            if (size > text.length) {
                break
            }
            for (start in 0..text.length - size) {
                val len = encodeUtf8(text = text, start = start, end = start + size, buf = buf, offset = 0)
                accumulator.addTo(key = bucketOf(buf = buf, length = len), delta = 1.0f)
            }
        }
    }

    private fun addWordNgrams(text: String, accumulator: IntFloatHashMap, buf: ByteArray) {
        val words = text.split(regex = WHITESPACE).filter { word -> word.isNotEmpty() }
        if (words.isEmpty()) {
            return
        }
        for (size in config.wordNgramMin..config.wordNgramMax) {
            if (size > words.size) {
                break
            }
            for (start in 0..words.size - size) {
                var pos = 0
                for (wi in start until start + size) {
                    if (wi > start) buf[pos++] = ' '.code.toByte()
                    pos += encodeUtf8(text = words[wi], start = 0, end = words[wi].length, buf = buf, offset = pos)
                }
                accumulator.addTo(key = bucketOf(buf = buf, length = pos), delta = 1.0f)
            }
        }
    }

    private fun bucketOf(buf: ByteArray, length: Int): Int {
        val hash = SipHash.hash(data = buf, length = length, key0 = config.key0, key1 = config.key1)
        val bucket = (hash % config.numFeatures).toInt()
        return if (bucket < 0) bucket + config.numFeatures else bucket
    }

    /**
     * Encodes [text][start..end) as UTF-8 bytes into [buf] starting at [offset].
     * Handles BMP code points (U+0000–U+FFFF) — sufficient for financial text.
     * Returns the number of bytes written.
     */
    // MagicNumber suppressed: every literal here is a fixed constant mandated by the UTF-8 spec.
    @Suppress("MagicNumber")
    private fun encodeUtf8(text: String, start: Int, end: Int, buf: ByteArray, offset: Int): Int {
        var pos = offset
        for (i in start until end) {
            val c = text[i].code
            when {
                c < 0x80 -> buf[pos++] = c.toByte()
                c < 0x800 -> {
                    buf[pos++] = (0xC0 or (c shr 6)).toByte()
                    buf[pos++] = (0x80 or (c and 0x3F)).toByte()
                }
                else -> {
                    buf[pos++] = (0xE0 or (c shr 12)).toByte()
                    buf[pos++] = (0x80 or ((c shr 6) and 0x3F)).toByte()
                    buf[pos++] = (0x80 or (c and 0x3F)).toByte()
                }
            }
        }
        return pos - offset
    }

    private companion object {
        val WHITESPACE = Regex(pattern = "\\s+")
    }
}
