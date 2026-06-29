package io.skein.examples.textrepair

import io.skein.text.application.BrokenWordRepairer
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.FrequencyModel
import io.skein.text.domain.PatternSignature
import io.skein.text.infrastructure.DefaultTextNormalizer

fun runTextRepairExample() {
    val corpus = listOf(
        "apartment rent monthly payment",
        "transfer bank account sepa",
        "apartment deposit first month",
        "monthly rent apartment payment",
        "bank account transfer wire",
    )

    val model = FrequencyModel(minKeepFrequency = 1)
    corpus.forEach { sentence -> model.learnAll(words = sentence.split(" ")) }

    println("=== FrequencyModel ===")
    println("known 'apartment' : ${model.isKnown(word = "apartment")} (freq=${model.frequency(word = "apartment")})")
    println("known 'xyzzy'     : ${model.isKnown(word = "xyzzy")}")

    val repairer = BrokenWordRepairer(frequencyModel = model)

    println()
    println("=== BrokenWordRepairer ===")
    val broken = "apart ment rent monthly pay ment"
    val repaired = repairer.repair(text = broken)
    println("input    : $broken")
    println("repaired : $repaired")

    println()
    println("=== Full pipeline: normalize → repair → tokenize → signature ===")
    val normalizer = DefaultTextNormalizer()
    val tokenizer = TypedTokenizer()

    val raw = "  Apart  MENT  Rent  12,50  Monthly  "
    val normalized = normalizer.normalize(raw = raw)
    val fixed = repairer.repair(text = normalized)
    val tokens = tokenizer.tokenize(text = fixed)
    val signature = PatternSignature.of(tokens = tokens)

    println("raw       : $raw")
    println("normalized: $normalized")
    println("repaired  : $fixed")
    println("tokens    : ${tokens.map { it.text }}")
    println("signature : ${signature.render()}")

    println()
    println("=== Serialize / deserialize round-trip ===")
    val serialized = model.serialize()
    val restored = FrequencyModel.deserialize(serialized = serialized)
    println("serialize lines : ${serialized.lines().filter { it.isNotBlank() }.size}")
    println("round-trip match: ${restored.knownWords() == model.knownWords()}")
}
