package io.skein.extract.domain

/** How many consecutive tokens a [PatternElement] may match — the "regex quantifiers" over token types. */
enum class QuantifierEnum {
    /** Exactly one token. */
    ONE,

    /** Zero or one token. */
    OPTIONAL,

    /** Zero or more tokens (greedy). */
    ZERO_OR_MORE,

    /** One or more tokens (greedy). */
    ONE_OR_MORE,
}
