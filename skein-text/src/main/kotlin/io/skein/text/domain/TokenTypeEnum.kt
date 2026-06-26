package io.skein.text.domain

/**
 * Structural type of a [Token].
 *
 * The type is a privacy-friendly feature: it captures the *shape* of a token
 * (word, number, date, ...) without retaining its content. Downstream modules
 * build pattern signatures from these types alone.
 */
enum class TokenTypeEnum {
    /** Pure alphabetic word, for example `apartment`. */
    WORD,

    /** Integer or grouped number without a decimal part, for example `1234` or `1.234`. */
    NUMERIC,

    /** Mix of letters and digits without other symbols, for example `AB12`. */
    ALPHANUMERIC,

    /** Calendar date such as `31.12.2024`, `31/12/2024` or `2024-12-31`. */
    DATE,

    /** Monetary amount such as `1.234,56`, `12,50` or `12.50`. */
    AMOUNT,

    /** Run consisting only of symbols/punctuation, for example `:` or `-->`. */
    SYMBOL,

    /** Word-like token that mixes letters/digits with symbols, for example `AIG-Life` or `CustomerNumber:`. */
    WORD_SYMBOL,
}
