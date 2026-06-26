package io.skein.text.application

import kotlin.math.min

/**
 * SymSpell-style index for fast "is this within edit distance k of any known word?" lookups.
 *
 * It precomputes, for every dictionary word, all strings reachable by deleting up to
 * [maxEditDistance] characters, mapping each such *deletion variant* back to the words that produced
 * it. A query is answered by generating the query's own deletion variants and intersecting — two
 * strings are within edit distance k iff they share a deletion variant — then confirming each
 * candidate with an exact bounded Levenshtein check. This turns the previous O(vocabulary) scan per
 * lookup into roughly O(query-length^k) map probes.
 */
class SymSpellIndex(words: Set<String>, private val maxEditDistance: Int) {

    private val wordsByDeletionVariant: Map<String, Set<String>> = buildIndex(words = words)

    /** Dictionary words within [maxEditDistance] edits of [query] (verified, not just candidates). */
    fun candidates(query: String): Set<String> {
        val matches = LinkedHashSet<String>()
        for (variant in deletionVariants(value = query)) {
            val words = wordsByDeletionVariant[variant] ?: emptySet()
            for (word in words) {
                if (isWithinEditDistance(left = query, right = word, limit = maxEditDistance)) {
                    matches.add(word)
                }
            }
        }
        return matches
    }

    private fun buildIndex(words: Set<String>): Map<String, Set<String>> {
        val index = HashMap<String, MutableSet<String>>()
        for (word in words) {
            for (variant in deletionVariants(value = word)) {
                index.getOrPut(key = variant) { HashSet() }.add(word)
            }
        }
        return index
    }

    /** [value] plus every string formed by deleting 1..[maxEditDistance] characters. */
    private fun deletionVariants(value: String): Set<String> {
        val variants = HashSet<String>()
        variants.add(value)
        var frontier = setOf(value)
        for (step in 0 until maxEditDistance) {
            val next = HashSet<String>()
            for (token in frontier) {
                for (index in token.indices) {
                    next.add(token.removeRange(startIndex = index, endIndex = index + 1))
                }
            }
            variants.addAll(next)
            frontier = next
        }
        return variants
    }

    /** Bounded Levenshtein: true when the edit distance between [left] and [right] is at most [limit]. */
    private fun isWithinEditDistance(left: String, right: String, limit: Int): Boolean {
        val previous = IntArray(right.length + 1) { it }
        val current = IntArray(right.length + 1)
        for (i in 1..left.length) {
            current[0] = i
            var rowMinimum = current[0]
            for (j in 1..right.length) {
                val substitutionCost = if (left[i - 1] == right[j - 1]) 0 else 1
                current[j] = min(
                    a = min(a = previous[j] + 1, b = current[j - 1] + 1),
                    b = previous[j - 1] + substitutionCost,
                )
                rowMinimum = min(a = rowMinimum, b = current[j])
            }
            if (rowMinimum > limit) {
                return false
            }
            System.arraycopy(current, 0, previous, 0, current.size)
        }
        return previous[right.length] <= limit
    }
}
