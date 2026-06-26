package io.skein.extract.domain

import io.skein.text.domain.TokenTypeEnum

/**
 * Extracts the value that follows a keyword anchor, for example `"CustomerNumber" → following ALPHANUMERIC`.
 *
 * The [anchor] is matched against token text ignoring case and surrounding punctuation (so
 * `CustomerNumber:` still matches `CustomerNumber`). When [targetType] is set, the first following token
 * of that type is taken; otherwise the immediately following token is taken.
 */
data class KeyAnchoredSlot(
    override val name: String,
    val anchor: String,
    val targetType: TokenTypeEnum? = null,
) : SlotDefinition
