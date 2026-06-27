package io.skein.extract.application

import io.skein.extract.domain.ExtractedField
import io.skein.extract.domain.ExtractionResult
import io.skein.extract.domain.KeyAnchoredSlot
import io.skein.extract.domain.PositionalSlot
import io.skein.extract.domain.RepeatingGroupSlot
import io.skein.extract.domain.SlotDefinition
import io.skein.extract.domain.SourceSpan
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token

/**
 * Fills [SlotDefinition]s from a token stream and returns the real extracted values. Positional and
 * key-anchored slots yield a single field; repeating-group slots yield one field per component per
 * occurrence, linked by `groupIndex`.
 */
class SlotExtractor(private val tokenizer: TypedTokenizer = TypedTokenizer()) {

    /** Tokenizes [text] then extracts. */
    fun extract(text: String, slots: List<SlotDefinition>): ExtractionResult {
        return extract(tokens = tokenizer.tokenize(text = text), slots = slots)
    }

    fun extract(tokens: List<Token>, slots: List<SlotDefinition>): ExtractionResult {
        val fields = ArrayList<ExtractedField>()
        for (slot in slots) {
            when (slot) {
                is PositionalSlot -> extractPositional(slot = slot, tokens = tokens, fields = fields)
                is KeyAnchoredSlot -> extractKeyAnchored(slot = slot, tokens = tokens, fields = fields)
                is RepeatingGroupSlot -> extractRepeating(slot = slot, tokens = tokens, fields = fields)
            }
        }
        return ExtractionResult(fields = fields)
    }

    private fun extractPositional(slot: PositionalSlot, tokens: List<Token>, fields: MutableList<ExtractedField>) {
        val token = tokens.getOrNull(index = slot.tokenIndex) ?: return
        fields.add(fieldOf(name = slot.name, token = token))
    }

    private fun extractKeyAnchored(slot: KeyAnchoredSlot, tokens: List<Token>, fields: MutableList<ExtractedField>) {
        val anchorIndex = tokens.indexOfFirst { token -> matchesAnchor(token = token, anchor = slot.anchor) }
        if (anchorIndex < 0) {
            return
        }
        val target = findTarget(tokens = tokens, anchorIndex = anchorIndex, slot = slot) ?: return
        fields.add(fieldOf(name = slot.name, token = target))
    }

    private fun findTarget(tokens: List<Token>, anchorIndex: Int, slot: KeyAnchoredSlot): Token? {
        val following = tokens.subList(anchorIndex + 1, tokens.size)
        if (slot.targetType == null) {
            return following.firstOrNull()
        }
        return following.firstOrNull { token -> token.type == slot.targetType }
    }

    private fun extractRepeating(slot: RepeatingGroupSlot, tokens: List<Token>, fields: MutableList<ExtractedField>) {
        val span = slot.components.size
        var position = 0
        var groupIndex = 0
        while (position + span <= tokens.size) {
            if (groupMatches(slot = slot, tokens = tokens, position = position)) {
                appendGroup(slot = slot, tokens = tokens, position = position, groupIndex = groupIndex, fields = fields)
                groupIndex++
                position += span
            } else {
                position++
            }
        }
    }

    private fun groupMatches(slot: RepeatingGroupSlot, tokens: List<Token>, position: Int): Boolean {
        return slot.components.withIndex().all { (offset, component) ->
            tokens[position + offset].type == component.type
        }
    }

    private fun appendGroup(
        slot: RepeatingGroupSlot,
        tokens: List<Token>,
        position: Int,
        groupIndex: Int,
        fields: MutableList<ExtractedField>,
    ) {
        slot.components.forEachIndexed { offset, component ->
            fields.add(
                fieldOf(name = component.name, token = tokens[position + offset], groupIndex = groupIndex),
            )
        }
    }

    private fun matchesAnchor(token: Token, anchor: String): Boolean {
        return token.text.trim { character -> !character.isLetterOrDigit() }.equals(other = anchor, ignoreCase = true)
    }

    private fun fieldOf(name: String, token: Token, groupIndex: Int? = null): ExtractedField {
        return ExtractedField(
            name = name,
            value = token.text,
            span = SourceSpan(startOffset = token.startOffset, endOffset = token.endOffset),
            groupIndex = groupIndex,
        )
    }
}
