package io.skein.extract.domain

/** Extracts the token at a fixed position, for example "the third token". */
data class PositionalSlot(
    override val name: String,
    val tokenIndex: Int,
) : SlotDefinition
