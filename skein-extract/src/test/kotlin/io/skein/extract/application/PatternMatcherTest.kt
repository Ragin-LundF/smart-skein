package io.skein.extract.application

import io.skein.extract.domain.TokenPattern
import io.skein.text.domain.TokenTypeEnum
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class PatternMatcherTest {

    private val matcher = PatternMatcher()

    @Test
    internal fun `finds a word followed by a date`() {
        val pattern = TokenPattern.of {
            type(TokenTypeEnum.WORD)
            type(TokenTypeEnum.DATE)
        }
        // tokens: booked(WORD) 2024-12-31(DATE) amount(WORD) 12.50(AMOUNT)
        val matches = matcher.findAll(text = "booked 2024-12-31 amount 12.50", pattern = pattern)
        assertEquals(expected = listOf(0 until 2), actual = matches)
    }

    @Test
    internal fun `optional element matches with or without the token`() {
        val pattern = TokenPattern.of {
            type(TokenTypeEnum.WORD)
            optional(TokenTypeEnum.SYMBOL)
            type(TokenTypeEnum.AMOUNT)
        }
        assertTrue(
            actual = matcher.findAll(text = "total : 12.50", pattern = pattern).isNotEmpty(),
            message = "matches with the symbol",
        )
        assertTrue(
            actual = matcher.findAll(text = "total 12.50", pattern = pattern).isNotEmpty(),
            message = "matches without the symbol",
        )
    }

    @Test
    internal fun `one or more is greedy across repeated tokens`() {
        val pattern = TokenPattern.of {
            type(TokenTypeEnum.WORD)
            oneOrMore(TokenTypeEnum.AMOUNT)
        }
        // sum 1.00 2.00 3.00 -> one match covering all four tokens
        assertEquals(
            expected = listOf(0 until 4),
            actual = matcher.findAll(text = "sum 1.00 2.00 3.00", pattern = pattern),
        )
    }

    @Test
    internal fun `matchesFully reports whole-stream matches`() {
        val tokenizer = io.skein.text.application.TypedTokenizer()
        val pattern = TokenPattern.of {
            type(TokenTypeEnum.WORD)
            type(TokenTypeEnum.AMOUNT)
        }
        assertTrue(actual = matcher.matchesFully(tokens = tokenizer.tokenize(text = "amount 12.50"), pattern = pattern))
        assertFalse(
            actual = matcher.matchesFully(tokens = tokenizer.tokenize(text = "amount 12.50 extra"), pattern = pattern),
        )
    }
}
