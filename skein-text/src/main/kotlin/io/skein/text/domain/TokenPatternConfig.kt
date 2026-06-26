package io.skein.text.domain

/**
 * Ordered typed-pattern rules used by the tokenizer to recognize and classify spans
 * like dates, amounts and numbers.
 *
 * Rules are evaluated most-specific first: the first full match wins for classification,
 * and the longest anchored match wins for boundary detection in
 * [TokenizationModeEnum.PUNCTUATION_AWARE]. Ordering therefore encodes priority — keep
 * `DATE` and `AMOUNT` ahead of `NUMERIC`, since both also look numeric.
 *
 * The locale-specific patterns live here so consumers can supply their own conventions
 * (US dates, dot-decimal amounts, ...) or add domain recognizers such as an IBAN matched
 * as a single token. The type label must be one of the existing [TokenTypeEnum] values.
 */
data class TokenPatternConfig(
    val typedRules: List<Pair<Regex, TokenTypeEnum>>,
) {
    companion object {
        /** German/European: dotted/slashed/ISO dates, comma-decimal amounts. The default. */
        val GERMAN = TokenPatternConfig(
            typedRules = listOf(
                Regex(pattern = "\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}|\\d{4}-\\d{2}-\\d{2}") to TokenTypeEnum.DATE,
                Regex(pattern = "\\d{1,3}(\\.\\d{3})*,\\d{2}|\\d+,\\d{2}|\\d+\\.\\d{2}") to TokenTypeEnum.AMOUNT,
                Regex(pattern = "\\d+([.,]\\d{3})*") to TokenTypeEnum.NUMERIC,
            ),
        )

        /** US/English: mm/dd/yyyy (or ISO) dates, dot-decimal comma-grouped amounts. */
        val US = TokenPatternConfig(
            typedRules = listOf(
                Regex(pattern = "\\d{1,2}/\\d{1,2}/\\d{2,4}|\\d{4}-\\d{2}-\\d{2}") to TokenTypeEnum.DATE,
                Regex(pattern = "\\d{1,3}(,\\d{3})*\\.\\d{2}|\\d+\\.\\d{2}") to TokenTypeEnum.AMOUNT,
                Regex(pattern = "\\d+(,\\d{3})*") to TokenTypeEnum.NUMERIC,
            ),
        )
    }
}
