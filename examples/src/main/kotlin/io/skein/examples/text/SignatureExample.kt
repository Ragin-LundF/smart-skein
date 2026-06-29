package io.skein.examples.text

import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.PatternSignature

fun runSignatureExample() {
    val tokenizer = TypedTokenizer()

    val similar1 = "AIG-Life 67,89 insurance premium"
    val similar2 = "Geico-Auto 120,00 insurance premium"
    val different1 = "rent apartment CustomerNumber AB123 monthly"
    val different2 = "salary 2024-03-01"

    fun signatureOf(text: String): PatternSignature {
        return PatternSignature.of(tokens = tokenizer.tokenize(text = text))
    }

    val sigSimilar1 = signatureOf(text = similar1)
    val sigSimilar2 = signatureOf(text = similar2)
    val sigDifferent1 = signatureOf(text = different1)
    val sigDifferent2 = signatureOf(text = different2)

    println("=== PatternSignature.of() ===")
    println()

    println("Similar pair — same layout:")
    println("  '$similar1'")
    println("  '$similar2'")
    println("  sig1 : ${sigSimilar1.render()}")
    println("  sig2 : ${sigSimilar2.render()}")
    println("  equal: ${sigSimilar1 == sigSimilar2}")

    println()
    println("Different layouts:")
    println("  '$different1'")
    println("    → ${sigDifferent1.render()}")
    println("  '$different2'")
    println("    → ${sigDifferent2.render()}")
    println("  equal: ${sigDifferent1 == sigDifferent2}")

    println()
    println("=== Cross-comparison matrix ===")
    val texts = listOf(similar1, similar2, different1, different2)
    val sigs = texts.map { text -> signatureOf(text = text) }

    texts.forEachIndexed { i, textA ->
        texts.forEachIndexed { j, textB ->
            if (j > i) {
                val match = sigs[i] == sigs[j]
                println("  [${i + 1}] vs [${j + 1}]  match=$match")
                if (!match) {
                    println("         [${i + 1}] ${sigs[i].render()}")
                    println("         [${j + 1}] ${sigs[j].render()}")
                }
            }
        }
    }
}
