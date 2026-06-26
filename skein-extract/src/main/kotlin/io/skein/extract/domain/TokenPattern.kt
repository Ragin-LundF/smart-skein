package io.skein.extract.domain

/**
 * A "regex over token types": an ordered list of [PatternElement]s matched against a token stream
 * by [io.skein.extract.application.PatternMatcher].
 *
 * Build via the [of] DSL:
 * ```
 * val pattern = TokenPattern.of {
 *     type(TokenTypeEnum.WORD)
 *     optional(TokenTypeEnum.SYMBOL)
 *     type(TokenTypeEnum.DATE)
 * }
 * ```
 */
class TokenPattern internal constructor(val elements: List<PatternElement>) {

    init {
        require(elements.isNotEmpty()) { "a pattern must have at least one element" }
    }

    companion object {
        fun of(block: TokenPatternBuilder.() -> Unit): TokenPattern {
            val builder = TokenPatternBuilder()
            builder.block()
            return builder.build()
        }
    }
}
