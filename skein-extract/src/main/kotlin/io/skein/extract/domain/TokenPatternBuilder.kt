package io.skein.extract.domain

import io.skein.text.domain.TokenTypeEnum

/** Mutable builder backing [TokenPattern.of]; one method per quantifier. */
class TokenPatternBuilder {

    private val elements = ArrayList<PatternElement>()

    fun type(type: TokenTypeEnum) {
        elements.add(PatternElement(type = type, quantifier = QuantifierEnum.ONE))
    }

    fun optional(type: TokenTypeEnum) {
        elements.add(PatternElement(type = type, quantifier = QuantifierEnum.OPTIONAL))
    }

    fun zeroOrMore(type: TokenTypeEnum) {
        elements.add(PatternElement(type = type, quantifier = QuantifierEnum.ZERO_OR_MORE))
    }

    fun oneOrMore(type: TokenTypeEnum) {
        elements.add(PatternElement(type = type, quantifier = QuantifierEnum.ONE_OR_MORE))
    }

    fun build(): TokenPattern {
        return TokenPattern(elements.toList())
    }
}
