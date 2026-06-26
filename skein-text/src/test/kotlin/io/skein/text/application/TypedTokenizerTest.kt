package io.skein.text.application

import io.skein.text.domain.TokenizationModeEnum
import io.skein.text.domain.TokenTypeEnum
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class TypedTokenizerTest {

    private val tokenizer = TypedTokenizer()

    private fun typesOf(text: String): List<TokenTypeEnum> {
        return tokenizer.tokenize(text = text).map { token -> token.type }
    }

    @Test
    internal fun `classifies a plain alphabetic word as WORD`() {
        assertEquals(expected = listOf(TokenTypeEnum.WORD), actual = typesOf(text = "apartment"))
    }

    @Test
    internal fun `classifies dotted slashed and iso dates as DATE`() {
        assertEquals(
            expected = listOf(TokenTypeEnum.DATE, TokenTypeEnum.DATE, TokenTypeEnum.DATE),
            actual = typesOf(text = "31.12.2024 31/12/24 2024-12-31"),
        )
    }

    @Test
    internal fun `classifies grouped and plain amounts as AMOUNT`() {
        assertEquals(
            expected = listOf(TokenTypeEnum.AMOUNT, TokenTypeEnum.AMOUNT, TokenTypeEnum.AMOUNT),
            actual = typesOf(text = "1.234,56 12,50 12.50"),
        )
    }

    @Test
    internal fun `classifies a plain integer as NUMERIC`() {
        assertEquals(expected = listOf(TokenTypeEnum.NUMERIC), actual = typesOf(text = "1234"))
    }

    @Test
    internal fun `classifies letters mixed with digits as ALPHANUMERIC`() {
        assertEquals(expected = listOf(TokenTypeEnum.ALPHANUMERIC), actual = typesOf(text = "AB12"))
    }

    @Test
    internal fun `classifies a symbol-only run as SYMBOL`() {
        assertEquals(expected = listOf(TokenTypeEnum.SYMBOL), actual = typesOf(text = "-->"))
    }

    @Test
    internal fun `classifies a word with internal symbols as WORD_SYMBOL`() {
        assertEquals(
            expected = listOf(TokenTypeEnum.WORD_SYMBOL, TokenTypeEnum.WORD_SYMBOL),
            actual = typesOf(text = "AIG-Life CustomerNumber:"),
        )
    }

    @Test
    internal fun `reports correct source offsets`() {
        val tokens = tokenizer.tokenize(text = "ab  cd")
        assertEquals(expected = 0, actual = tokens[0].startOffset)
        assertEquals(expected = 2, actual = tokens[0].endOffset)
        assertEquals(expected = 4, actual = tokens[1].startOffset)
        assertEquals(expected = 6, actual = tokens[1].endOffset)
        assertEquals(expected = "cd", actual = tokens[1].text)
    }

    @Test
    internal fun `returns no tokens for blank input`() {
        assertTrue(actual = tokenizer.tokenize(text = "   \t\n ").isEmpty())
    }

    @Test
    internal fun `punctuation-aware mode splits trailing punctuation off words`() {
        val aware = TypedTokenizer(mode = TokenizationModeEnum.PUNCTUATION_AWARE)
        val tokens = aware.tokenize(text = "CustomerNumber: AB12")
        assertEquals(
            expected = listOf(TokenTypeEnum.WORD, TokenTypeEnum.SYMBOL, TokenTypeEnum.ALPHANUMERIC),
            actual = tokens.map { token -> token.type },
        )
        assertEquals(expected = "CustomerNumber", actual = tokens[0].text)
        assertEquals(expected = ":", actual = tokens[1].text)
    }

    @Test
    internal fun `punctuation-aware mode keeps dates and amounts intact`() {
        val aware = TypedTokenizer(mode = TokenizationModeEnum.PUNCTUATION_AWARE)
        assertEquals(
            expected = listOf(TokenTypeEnum.DATE, TokenTypeEnum.AMOUNT),
            actual = aware.tokenize(text = "2024-12-31 1.234,56").map { token -> token.type },
        )
    }
}
