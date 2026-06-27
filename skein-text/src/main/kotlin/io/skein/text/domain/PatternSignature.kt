package io.skein.text.domain

/**
 * The ordered sequence of token types of a text fragment — a privacy-preserving
 * structural fingerprint such as `<word> <date> <numeric>`.
 *
 * It carries only structure, never content, so it can be shared and compared
 * across records without exposing personal data.
 */
@JvmInline
value class PatternSignature(val types: List<TokenTypeEnum>) {

    /** Human-readable rendering such as `<word> <date> <numeric>`. */
    fun render(): String {
        return types.joinToString(separator = " ") { type -> "<${type.name.lowercase()}>" }
    }

    companion object {
        /** Derives the signature from already-tokenized input, preserving order. */
        fun of(tokens: List<Token>): PatternSignature {
            return PatternSignature(types = tokens.map { token -> token.type })
        }
    }
}
