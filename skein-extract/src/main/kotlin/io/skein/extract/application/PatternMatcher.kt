package io.skein.extract.application

import io.skein.extract.domain.PatternElement
import io.skein.extract.domain.QuantifierEnum
import io.skein.extract.domain.TokenPattern
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token

/**
 * Matches a [TokenPattern] (regex over token types) against a token stream using greedy matching
 * with backtracking. [findAll] returns the leftmost, non-overlapping matches as token-index ranges.
 */
class PatternMatcher(private val tokenizer: TypedTokenizer = TypedTokenizer()) {

    /** Tokenizes [text] then matches; convenience over [findAll] with an explicit token list. */
    fun findAll(text: String, pattern: TokenPattern): List<IntRange> {
        return findAll(tokenizer.tokenize(text), pattern)
    }

    /** Leftmost, non-overlapping matches as inclusive token-index ranges. */
    fun findAll(tokens: List<Token>, pattern: TokenPattern): List<IntRange> {
        val matches = ArrayList<IntRange>()
        var start = 0
        while (start < tokens.size) {
            val end = matchFrom(tokens, pattern.elements, start, 0)
            if (end > start) {
                matches.add(start until end)
                start = end
            } else {
                start++
            }
        }
        return matches
    }

    /** Whether [pattern] matches the entire token list. */
    fun matchesFully(tokens: List<Token>, pattern: TokenPattern): Boolean {
        return matchFrom(tokens, pattern.elements, 0, 0) == tokens.size
    }

    /** Returns the end token index of a successful match starting at [tokenIndex], or -1. */
    private fun matchFrom(
        tokens: List<Token>,
        elements: List<PatternElement>,
        tokenIndex: Int,
        elementIndex: Int,
    ): Int {
        if (elementIndex == elements.size) {
            return tokenIndex
        }
        val element = elements[elementIndex]
        return when (element.quantifier) {
            QuantifierEnum.ONE -> matchOne(tokens, elements, tokenIndex, elementIndex)
            QuantifierEnum.OPTIONAL -> matchOptional(tokens, elements, tokenIndex, elementIndex)
            QuantifierEnum.ZERO_OR_MORE -> matchRepeated(tokens, elements, tokenIndex, elementIndex, minCount = 0)
            QuantifierEnum.ONE_OR_MORE -> matchRepeated(tokens, elements, tokenIndex, elementIndex, minCount = 1)
        }
    }

    private fun matchOne(tokens: List<Token>, elements: List<PatternElement>, tokenIndex: Int, elementIndex: Int): Int {
        if (tokenIndex < tokens.size && tokens[tokenIndex].type == elements[elementIndex].type) {
            return matchFrom(tokens, elements, tokenIndex + 1, elementIndex + 1)
        }
        return NO_MATCH
    }

    private fun matchOptional(
        tokens: List<Token>,
        elements: List<PatternElement>,
        tokenIndex: Int,
        elementIndex: Int,
    ): Int {
        if (tokenIndex < tokens.size && tokens[tokenIndex].type == elements[elementIndex].type) {
            val matched = matchFrom(tokens, elements, tokenIndex + 1, elementIndex + 1)
            if (matched >= 0) {
                return matched
            }
        }
        return matchFrom(tokens, elements, tokenIndex, elementIndex + 1)
    }

    private fun matchRepeated(
        tokens: List<Token>,
        elements: List<PatternElement>,
        tokenIndex: Int,
        elementIndex: Int,
        minCount: Int,
    ): Int {
        val type = elements[elementIndex].type
        var available = 0
        while (tokenIndex + available < tokens.size && tokens[tokenIndex + available].type == type) {
            available++
        }
        var take = available
        while (take >= minCount) {
            val matched = matchFrom(tokens, elements, tokenIndex + take, elementIndex + 1)
            if (matched >= 0) {
                return matched
            }
            take--
        }
        return NO_MATCH
    }

    private companion object {
        const val NO_MATCH = -1
    }
}
