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

    fun vectorize(text: String): FeatureVector {
        val normalized = normalizer.normalize(raw = text)
        val accumulator = scratch.get()
        accumulator.clear()
        addCharNgrams(text = normalized, accumulator = accumulator)
        addWordNgrams(text = normalized, accumulator = accumulator)
        val (indices, values) = accumulator.sortedKeysAndValues()
        return FeatureVector(indices = indices, values = values)
    }

    private fun addCharNgrams(text: String, accumulator: IntFloatHashMap) {
        if (text.isEmpty()) {
            return
        }
        for (size in config.charNgramMin..config.charNgramMax) {
            if (size > text.length) {
                break
            }
            for (start in 0..text.length - size) {
                accumulate(ngram = text.substring(start, start + size), accumulator = accumulator)
            }
        }
    }

    private fun addWordNgrams(text: String, accumulator: IntFloatHashMap) {
        val words = text.split(regex = WHITESPACE).filter { word -> word.isNotEmpty() }
        if (words.isEmpty()) {
            return
        }
        for (size in config.wordNgramMin..config.wordNgramMax) {
            if (size > words.size) {
                break
            }
            for (start in 0..words.size - size) {
                accumulate(
                    ngram = words.subList(start, start + size).joinToString(separator = " "),
                    accumulator = accumulator,
                )
            }
        }
    }

    private fun accumulate(ngram: String, accumulator: IntFloatHashMap) {
        accumulator.addTo(key = indexOf(ngram), delta = 1.0f)
    }

    private fun indexOf(ngram: String): Int {
        val hash = SipHash.hash(data = ngram.toByteArray(Charsets.UTF_8), key0 = config.key0, key1 = config.key1)
        val bucket = (hash % config.numFeatures).toInt()
        return if (bucket < 0) bucket + config.numFeatures else bucket
    }

    private companion object {
        val WHITESPACE = Regex(pattern = "\\s+")
    }
}
