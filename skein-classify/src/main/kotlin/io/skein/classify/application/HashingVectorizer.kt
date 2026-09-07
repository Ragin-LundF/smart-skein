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

    // Reusable word-boundary buffer: interleaved (start, end) pairs for up to 128 words per text.
    // Normalized text (single-space-separated, trimmed) rarely exceeds ~30 words in financial records.
    private val wordBounds = ThreadLocal.withInitial { IntArray(256) }

    fun vectorize(text: String): FeatureVector {
        val normalized = normalizer.normalize(raw = text)
        val accumulator = scratch.get()
        val buf = encBuf.get()
        accumulator.clear()
        forEachCharNgram(text = normalized, buf = buf) { bucket, _, _ ->
            accumulator.addTo(key = bucket, delta = 1.0f)
        }
        forEachWordNgram(text = normalized, buf = buf) { bucket, _, _ ->
            accumulator.addTo(key = bucket, delta = 1.0f)
        }
        val (indices, values) = accumulator.sortedKeysAndValues()
        return FeatureVector(indices = indices, values = values)
    }

    /**
     * Maps each feature bucket of [text] back to one representative n-gram of [text].
     *
     * **Opt-in and privacy-relevant.** This builds nothing and stores nothing: the mapping is
     * re-derived on the call from text the caller already holds, so it reveals nothing that a caller
     * holding both the hashing key and the record could not compute for itself by calling
     * [vectorize] on candidate n-grams. The returned strings are source text — see
     * [io.skein.classify.domain.AttributionModeEnum].
     *
     * ponytail: the first n-gram to reach a bucket wins, so colliding n-grams are not reported and
     * the result is a representative rather than a complete pre-image. Upgrade path: return
     * `Map<Int, List<String>>`.
     */
    fun ngramsByBucket(text: String): Map<Int, String> {
        val normalized = normalizer.normalize(raw = text)
        val buf = encBuf.get()
        val byBucket = HashMap<Int, String>()
        forEachCharNgram(text = normalized, buf = buf) { bucket, bytes, length ->
            byBucket.putIfAbsent(bucket, String(bytes, 0, length, Charsets.UTF_8))
        }
        forEachWordNgram(text = normalized, buf = buf) { bucket, bytes, length ->
            byBucket.putIfAbsent(bucket, String(bytes, 0, length, Charsets.UTF_8))
        }
        return byBucket
    }

    /**
     * Enumerates every character n-gram of [text], invoking [emit] with the bucket, the shared
     * encode buffer and the byte length.
     *
     * Inline with a crossinline sink so [vectorize] keeps the exact loop body it had before this
     * was shared — the alternative, passing an optional collector into the hot path, would add a
     * field read and a branch to the innermost loop of the most benchmark-sensitive method here for
     * the sake of a feature almost no call uses.
     */
    private inline fun forEachCharNgram(text: String, buf: ByteArray, emit: (Int, ByteArray, Int) -> Unit) {
        if (text.isEmpty()) {
            return
        }
        for (size in config.charNgramMin..config.charNgramMax) {
            if (size > text.length) {
                break
            }
            for (start in 0..text.length - size) {
                val len = encodeUtf8(text = text, start = start, end = start + size, buf = buf, offset = 0)
                emit(bucketOf(buf = buf, length = len), buf, len)
            }
        }
    }

    /** Enumerates every word n-gram of [text]. See [forEachCharNgram] for why this is inline. */
    private inline fun forEachWordNgram(text: String, buf: ByteArray, emit: (Int, ByteArray, Int) -> Unit) {
        val bounds = wordBounds.get()
        var wordCount = 0
        var wordStart = -1
        for (i in text.indices) {
            if (text[i] == ' ') {
                if (wordStart >= 0) {
                    bounds[wordCount * 2] = wordStart
                    bounds[wordCount * 2 + 1] = i
                    wordCount++
                    wordStart = -1
                }
            } else if (wordStart < 0) {
                wordStart = i
            }
        }
        if (wordStart >= 0) {
            bounds[wordCount * 2] = wordStart
            bounds[wordCount * 2 + 1] = text.length
            wordCount++
        }
        if (wordCount == 0) {
            return
        }
        for (size in config.wordNgramMin..config.wordNgramMax) {
            if (size > wordCount) {
                break
            }
            for (start in 0..wordCount - size) {
                var pos = 0
                for (wi in start until start + size) {
                    if (wi > start) buf[pos++] = ' '.code.toByte()
                    val wStart = bounds[wi * 2]
                    val wEnd = bounds[wi * 2 + 1]
                    pos += encodeUtf8(text = text, start = wStart, end = wEnd, buf = buf, offset = pos)
                }
                emit(bucketOf(buf = buf, length = pos), buf, pos)
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

}
