package io.skein.text.application

import io.skein.text.domain.Token
import io.skein.text.domain.TokenizationModeEnum
import io.skein.text.domain.TokenTypeEnum

/**
 * Splits text into typed [Token]s, classifying each into a [TokenTypeEnum].
 *
 * The classification is heuristic and tuned for European (notably German) financial text:
 * dates use `.` / `-` / `/` separators and amounts use a comma decimal with optional `.`
 * thousands grouping. These regex patterns are the calibration knobs — real-world data
 * may need locale-specific tuning.
 *
 * [mode] selects the splitting strategy. [TokenizationModeEnum.WHITESPACE] (default) splits only on
 * whitespace and keeps tokens like `1,234.56` and `AIG-Life` intact. [TokenizationModeEnum.PUNCTUATION_AWARE]
 * splits trailing/leading punctuation into separate `SYMBOL` tokens while keeping dates, amounts and
 * plain numbers whole — useful when a consumer needs finer-grained tokens.
 */
class TypedTokenizer(private val mode: TokenizationModeEnum = TokenizationModeEnum.WHITESPACE) {

    fun tokenize(text: String): List<Token> {
        return when (mode) {
            TokenizationModeEnum.WHITESPACE -> tokenizeWhitespace(text = text)
            TokenizationModeEnum.PUNCTUATION_AWARE -> tokenizePunctuationAware(text = text)
        }
    }

    private fun tokenizeWhitespace(text: String): List<Token> {
        val tokens = ArrayList<Token>()
        var index = 0
        val length = text.length
        while (index < length) {
            if (text[index].isWhitespace()) {
                index++
                continue
            }
            val start = index
            while (index < length && !text[index].isWhitespace()) {
                index++
            }
            val raw = text.substring(startIndex = start, endIndex = index)
            tokens.add(Token(text = raw, type = classify(raw = raw), startOffset = start, endOffset = index))
        }
        return tokens
    }

    private fun tokenizePunctuationAware(text: String): List<Token> {
        val tokens = ArrayList<Token>()
        var index = 0
        while (index < text.length) {
            if (text[index].isWhitespace()) {
                index++
                continue
            }
            index = emitNextToken(text = text, start = index, tokens = tokens)
        }
        return tokens
    }

    private fun emitNextToken(text: String, start: Int, tokens: MutableList<Token>): Int {
        val typedEnd = matchTypedAt(text = text, start = start)
        if (typedEnd > start) {
            addToken(
                tokens = tokens,
                text = text,
                start = start,
                end = typedEnd,
                type = classify(raw = text.substring(startIndex = start, endIndex = typedEnd)),
            )
            return typedEnd
        }
        if (text[start].isLetterOrDigit()) {
            var end = start
            while (end < text.length && text[end].isLetterOrDigit()) {
                end++
            }
            addToken(
                tokens = tokens,
                text = text,
                start = start,
                end = end,
                type = classifyAlphanumericRun(run = text.substring(startIndex = start, endIndex = end)),
            )
            return end
        }
        var end = start
        while (end < text.length && !text[end].isWhitespace() && !text[end].isLetterOrDigit()) {
            end++
        }
        addToken(tokens = tokens, text = text, start = start, end = end, type = TokenTypeEnum.SYMBOL)
        return end
    }

    /** End index (exclusive) of the longest DATE/AMOUNT/NUMERIC match anchored at [start], else [start]. */
    private fun matchTypedAt(text: String, start: Int): Int {
        var bestEnd = start
        for (pattern in TYPED_PATTERNS) {
            val match = pattern.matchAt(input = text, index = start)
            if (match != null && match.range.last + 1 > bestEnd) {
                bestEnd = match.range.last + 1
            }
        }
        return bestEnd
    }

    private fun addToken(tokens: MutableList<Token>, text: String, start: Int, end: Int, type: TokenTypeEnum) {
        tokens.add(
            Token(
                text = text.substring(startIndex = start, endIndex = end),
                type = type,
                startOffset = start,
                endOffset = end,
            ),
        )
    }

    private fun classifyAlphanumericRun(run: String): TokenTypeEnum {
        return when {
            LETTERS.matches(input = run) -> TokenTypeEnum.WORD
            run.all { character -> character.isDigit() } -> TokenTypeEnum.NUMERIC
            else -> TokenTypeEnum.ALPHANUMERIC
        }
    }

    private fun classify(raw: String): TokenTypeEnum {
        // Ordered most-specific first: a date also looks numeric, an amount also looks numeric.
        return when {
            DATE.matches(input = raw) -> TokenTypeEnum.DATE
            AMOUNT.matches(input = raw) -> TokenTypeEnum.AMOUNT
            NUMERIC.matches(input = raw) -> TokenTypeEnum.NUMERIC
            LETTERS.matches(input = raw) -> TokenTypeEnum.WORD
            ALPHANUMERIC.matches(input = raw) -> TokenTypeEnum.ALPHANUMERIC
            SYMBOLS.matches(input = raw) -> TokenTypeEnum.SYMBOL
            else -> TokenTypeEnum.WORD_SYMBOL
        }
    }

    private companion object {
        val DATE = Regex(pattern = "\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}|\\d{4}-\\d{2}-\\d{2}")
        val AMOUNT = Regex(pattern = "\\d{1,3}(\\.\\d{3})*,\\d{2}|\\d+,\\d{2}|\\d+\\.\\d{2}")
        val NUMERIC = Regex(pattern = "\\d+([.,]\\d{3})*")
        val LETTERS = Regex(pattern = "\\p{L}+")
        val ALPHANUMERIC = Regex(pattern = "[\\p{L}\\d]+")
        val SYMBOLS = Regex(pattern = "[^\\p{L}\\d\\s]+")
        val TYPED_PATTERNS = listOf(DATE, AMOUNT, NUMERIC)
    }
}
