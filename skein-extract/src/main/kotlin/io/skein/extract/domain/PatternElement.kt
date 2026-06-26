package io.skein.extract.domain

import io.skein.text.domain.TokenTypeEnum

/** One element of a [TokenPattern]: a token [type] to match, with a [quantifier]. */
data class PatternElement(
    val type: TokenTypeEnum,
    val quantifier: QuantifierEnum = QuantifierEnum.ONE,
)
