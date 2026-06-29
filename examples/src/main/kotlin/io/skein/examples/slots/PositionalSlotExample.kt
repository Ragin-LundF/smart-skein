package io.skein.examples.slots

import io.skein.extract.application.SlotExtractor
import io.skein.extract.domain.KeyAnchoredSlot
import io.skein.extract.domain.PositionalSlot
import io.skein.text.domain.TokenTypeEnum

fun runPositionalSlotExample() {
    val extractor = SlotExtractor()

    println("=== PositionalSlot: extract token at fixed index ===")
    val text = "SEPA transfer DE89370400440532013000 monthly"
    println("text: $text")

    val ibanSlot = PositionalSlot(name = "iban", tokenIndex = 2)
    val result = extractor.extract(text = text, slots = listOf(ibanSlot))
    println("token[2] (iban)  : ${result.first(name = "iban")?.value}")

    val wordSlot = PositionalSlot(name = "verb", tokenIndex = 1)
    val verbResult = extractor.extract(text = text, slots = listOf(wordSlot))
    println("token[1] (verb)  : ${verbResult.first(name = "verb")?.value}")

    println()
    println("=== PositionalSlot: out-of-bounds returns null ===")
    val shortText = "salary payout"
    val farSlot = PositionalSlot(name = "missing", tokenIndex = 10)
    val missing = extractor.extract(text = shortText, slots = listOf(farSlot))
    println("text: '$shortText'  token[10]: ${missing.first(name = "missing")}")

    println()
    println("=== KeyAnchoredSlot: contrast — follow keyword to value ===")
    val anchoredText = "rent apartment CustomerNumber AB123 monthly"
    println("text: $anchoredText")

    val customerSlot = KeyAnchoredSlot(
        name = "customer",
        anchor = "CustomerNumber",
        targetType = TokenTypeEnum.ALPHANUMERIC,
    )
    val anchoredResult = extractor.extract(text = anchoredText, slots = listOf(customerSlot))
    println("KeyAnchoredSlot 'CustomerNumber' → ALPHANUMERIC : ${anchoredResult.first(name = "customer")?.value}")

    println()
    println("=== KeyAnchoredSlot: anchor absent returns null ===")
    val noAnchorText = "rent apartment monthly"
    val noAnchorResult = extractor.extract(text = noAnchorText, slots = listOf(customerSlot))
    println("text: '$noAnchorText'  result: ${noAnchorResult.first(name = "customer")}")

    println()
    println("=== PositionalSlot vs KeyAnchoredSlot on same text ===")
    val both = listOf(
        PositionalSlot(name = "positional_3", tokenIndex = 3),
        KeyAnchoredSlot(name = "anchored", anchor = "CustomerNumber", targetType = TokenTypeEnum.ALPHANUMERIC),
    )
    val combined = extractor.extract(text = anchoredText, slots = both)
    println("positional token[3] : ${combined.first(name = "positional_3")?.value}")
    println("anchored value      : ${combined.first(name = "anchored")?.value}")
}
