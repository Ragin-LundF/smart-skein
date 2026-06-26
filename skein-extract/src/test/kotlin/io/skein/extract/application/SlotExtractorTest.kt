package io.skein.extract.application

import io.skein.extract.domain.GroupComponent
import io.skein.extract.domain.KeyAnchoredSlot
import io.skein.extract.domain.PositionalSlot
import io.skein.extract.domain.RepeatingGroupSlot
import io.skein.text.domain.TokenTypeEnum
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

internal class SlotExtractorTest {

    private val extractor = SlotExtractor()

    @Test
    internal fun `extracts a positional slot`() {
        val result = extractor.extract(
            text = "alpha beta gamma",
            slots = listOf(PositionalSlot(name = "second", tokenIndex = 1)),
        )
        assertEquals(expected = "beta", actual = result.first(name = "second")?.value)
    }

    @Test
    internal fun `extracts a key-anchored value past trailing punctuation`() {
        val slot = KeyAnchoredSlot(
            name = "customer",
            anchor = "CustomerNumber",
            targetType = TokenTypeEnum.ALPHANUMERIC,
        )
        // "CustomerNumber:" tokenizes as WORD_SYMBOL; the anchor still matches after stripping ':'.
        val result = extractor.extract(text = "CustomerNumber: AB12345 more info", slots = listOf(slot))
        assertEquals(expected = "AB12345", actual = result.first(name = "customer")?.value)
    }

    @Test
    internal fun `records the source span of an extracted value`() {
        val result = extractor.extract(
            text = "alpha beta",
            slots = listOf(PositionalSlot(name = "first", tokenIndex = 0)),
        )
        val span = result.first(name = "first")?.span
        assertEquals(expected = 0, actual = span?.startOffset)
        assertEquals(expected = 5, actual = span?.endOffset)
    }

    @Test
    internal fun `extracts repeating insurer and amount pairs linked by group index`() {
        val slot = RepeatingGroupSlot(
            name = "policies",
            components = listOf(
                GroupComponent(name = "insurer", type = TokenTypeEnum.WORD_SYMBOL),
                GroupComponent(name = "amount", type = TokenTypeEnum.AMOUNT),
            ),
        )
        val result = extractor.extract(text = "AIG-Life 67.89 Geico-Auto 120.00", slots = listOf(slot))

        assertEquals(expected = listOf("AIG-Life", "Geico-Auto"), actual = result.valuesOf(name = "insurer"))
        assertEquals(expected = listOf("67.89", "120.00"), actual = result.valuesOf(name = "amount"))
        // First group pairs AIG-Life with 67.89.
        val firstGroup = result.fields.filter { field -> field.groupIndex == 0 }
        assertTrue(actual = firstGroup.any { it.name == "insurer" && it.value == "AIG-Life" })
        assertTrue(actual = firstGroup.any { it.name == "amount" && it.value == "67.89" })
    }

    @Test
    internal fun `returns nothing for a missing anchor`() {
        val slot = KeyAnchoredSlot(name = "customer", anchor = "ContractNumber")
        val result = extractor.extract(text = "no such anchor here", slots = listOf(slot))
        assertNull(actual = result.first(name = "customer"))
    }
}
