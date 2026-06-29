package io.skein.examples.tokenization

import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.TokenizationModeEnum

fun runTokenizationModesExample() {
    val text = "AIG-Life 123,45 CustomerNumber:"

    val whitespace = TypedTokenizer(mode = TokenizationModeEnum.WHITESPACE)
    val punctuation = TypedTokenizer(mode = TokenizationModeEnum.PUNCTUATION_AWARE)

    val wsTokens = whitespace.tokenize(text = text)
    val paTokens = punctuation.tokenize(text = text)

    println("=== Input ===")
    println("  $text")
    println()

    println("=== WHITESPACE mode (${wsTokens.size} tokens) ===")
    wsTokens.forEach { token ->
        println("  '${token.text}'  type=${token.type}  [${token.startOffset}..${token.endOffset})")
    }

    println()
    println("=== PUNCTUATION_AWARE mode (${paTokens.size} tokens) ===")
    paTokens.forEach { token ->
        println("  '${token.text}'  type=${token.type}  [${token.startOffset}..${token.endOffset})")
    }

    println()
    println("=== Diff ===")
    println("  WHITESPACE keeps 'AIG-Life' and 'CustomerNumber:' as single tokens.")
    println("  PUNCTUATION_AWARE splits trailing ':' from 'CustomerNumber' into a SYMBOL.")
    println("  Both modes keep '123,45' intact as AMOUNT.")
}
