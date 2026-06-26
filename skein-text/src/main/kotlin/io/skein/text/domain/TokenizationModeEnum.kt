package io.skein.text.domain

/**
 * How [io.skein.text.application.TypedTokenizer] splits text into tokens.
 */
enum class TokenizationModeEnum {
    /**
     * Splits only on whitespace; punctuation stays attached, so `R+V` and `CustomerNumber:` remain
     * single [TokenTypeEnum.WORD_SYMBOL] tokens. The default — preserves structural tokens like
     * insurer codes and key:value pairs.
     */
    WHITESPACE,

    /**
     * Splits trailing/leading punctuation into separate [TokenTypeEnum.SYMBOL] tokens while keeping
     * dates, amounts and plain numbers intact. `CustomerNumber:` becomes `WORD` + `SYMBOL`.
     */
    PUNCTUATION_AWARE,
}
