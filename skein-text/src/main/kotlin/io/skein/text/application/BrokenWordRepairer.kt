package io.skein.text.application

import io.skein.text.domain.FrequencyModel
import kotlin.math.ln

/**
 * Repairs wrongly split words such as `"apart ment" → "apartment"`.
 *
 * It re-segments a whitespace-separated fragment sequence with dynamic programming over word
 * boundaries: consecutive fragments may be merged when the concatenation is a known word
 * (per [FrequencyModel]) or within [maxEditDistance] of one (typo tolerance). Long compound words
 * are handled naturally because the frequency model learns them as single long words.
 *
 * The objective rewards covering the text with as many high-confidence known words as possible
 * while strongly penalizing merges that would produce non-words, so genuinely separate words
 * are left untouched. Content is never rewritten — a near-known fragment keeps its original
 * spelling; the model only decides where word boundaries belong.
 */
class BrokenWordRepairer(
    private val frequencyModel: FrequencyModel,
    private val maxFragmentsPerWord: Int = DEFAULT_MAX_FRAGMENTS_PER_WORD,
    private val maxEditDistance: Int = DEFAULT_MAX_EDIT_DISTANCE,
) {

    private var cachedIndex: SymSpellIndex? = null
    private var cachedVocabularySize: Int = -1

    fun repair(text: String): String {
        val fragments = text.split(regex = WHITESPACE).filter { fragment -> fragment.isNotEmpty() }
        if (fragments.isEmpty()) {
            return ""
        }
        return resegment(fragments = fragments).joinToString(separator = " ")
    }

    private fun resegment(fragments: List<String>): List<String> {
        val count = fragments.size
        val bestScore = DoubleArray(count + 1) { Double.NEGATIVE_INFINITY }
        val wordStart = IntArray(count + 1) { -1 }
        bestScore[0] = 0.0
        for (end in 1..count) {
            val earliestStart = maxOf(a = 0, b = end - maxFragmentsPerWord)
            for (start in earliestStart until end) {
                if (bestScore[start] == Double.NEGATIVE_INFINITY) {
                    continue
                }
                val candidate = concat(fragments = fragments, start = start, end = end)
                val score = bestScore[start] + scoreOf(candidate = candidate, spanSize = end - start)
                if (score > bestScore[end]) {
                    bestScore[end] = score
                    wordStart[end] = start
                }
            }
        }
        return reconstruct(fragments = fragments, wordStart = wordStart, count = count)
    }

    private fun scoreOf(candidate: String, spanSize: Int): Double {
        val frequency = frequencyModel.frequency(word = candidate)
        if (frequency > 0) {
            return KNOWN_WORD_REWARD + ln(x = frequency.toDouble())
        }
        if (candidate.length >= MIN_TYPO_LENGTH && nearKnown(candidate = candidate)) {
            return NEAR_KNOWN_REWARD
        }
        // An unknown single fragment is kept as-is; merging fragments into a non-word is rejected.
        return if (spanSize == 1) UNKNOWN_FRAGMENT_PENALTY else spanSize * UNKNOWN_MERGE_PENALTY
    }

    private fun nearKnown(candidate: String): Boolean {
        return symSpellIndex().candidates(query = candidate).isNotEmpty()
    }

    /**
     * Returns a [SymSpellIndex] over the current vocabulary, rebuilding only when the vocabulary
     * grows. The frequency model is append-only (counts never decrease, words only cross the
     * threshold upward), so a size change is a reliable staleness signal.
     */
    private fun symSpellIndex(): SymSpellIndex {
        val known = frequencyModel.knownWords()
        val existing = cachedIndex
        if (existing != null && known.size == cachedVocabularySize) {
            return existing
        }
        val rebuilt = SymSpellIndex(words = known, maxEditDistance = maxEditDistance)
        cachedIndex = rebuilt
        cachedVocabularySize = known.size
        return rebuilt
    }

    private fun reconstruct(fragments: List<String>, wordStart: IntArray, count: Int): List<String> {
        val words = ArrayList<String>()
        var end = count
        while (end > 0) {
            val start = wordStart[end]
            words.add(concat(fragments, start, end))
            end = start
        }
        words.reverse()
        return words
    }

    private fun concat(fragments: List<String>, start: Int, end: Int): String {
        val builder = StringBuilder()
        for (index in start until end) {
            builder.append(fragments[index])
        }
        return builder.toString()
    }

    companion object {
        const val DEFAULT_MAX_FRAGMENTS_PER_WORD = 4
        const val DEFAULT_MAX_EDIT_DISTANCE = 1
        private const val MIN_TYPO_LENGTH = 4
        private const val KNOWN_WORD_REWARD = 10.0
        private const val NEAR_KNOWN_REWARD = 8.0
        private const val UNKNOWN_FRAGMENT_PENALTY = -1.0
        private const val UNKNOWN_MERGE_PENALTY = -1000.0
        private val WHITESPACE = Regex(pattern = "\\s+")
    }
}
