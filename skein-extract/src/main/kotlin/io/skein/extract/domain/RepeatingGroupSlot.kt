package io.skein.extract.domain

/**
 * Extracts a repeating run of consecutive component tokens, for example each
 * `WORD_SYMBOL + AMOUNT → (insurer, amount)` pair in a statement. Every match emits one field per
 * component, sharing an occurrence `groupIndex` so the parts of one group stay associated.
 */
data class RepeatingGroupSlot(
    override val name: String,
    val components: List<GroupComponent>,
) : SlotDefinition {

    init {
        require(components.isNotEmpty()) { "a repeating group needs at least one component" }
    }
}
