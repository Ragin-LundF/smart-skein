package io.skein.extract.domain

import io.skein.text.domain.TokenTypeEnum

/** One slot inside a [RepeatingGroupSlot]: a named value of a given token [type]. */
data class GroupComponent(val name: String, val type: TokenTypeEnum)
