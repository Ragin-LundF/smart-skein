package io.skein.examples.patternmatching

import io.skein.extract.application.PatternMatcher
import io.skein.extract.domain.TokenPattern
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.TokenTypeEnum

fun runTokenPatternExample() {
    val tokenizer = TypedTokenizer()
    val matcher = PatternMatcher(tokenizer = tokenizer)

    val wordAmount = TokenPattern.of {
        type(type = TokenTypeEnum.WORD_SYMBOL)
        type(type = TokenTypeEnum.AMOUNT)
    }

    println("=== findAll: WORD_SYMBOL AMOUNT in a multi-occurrence string ===")
    val text = "AIG-Life 67,89 Geico-Auto 120,00 insurance premium"
    val tokens = tokenizer.tokenize(text = text)
    println("tokens  : ${tokens.map { "${it.text}(${it.type})" }}")

    val ranges = matcher.findAll(tokens = tokens, pattern = wordAmount)
    println("matches : ${ranges.size}")
    ranges.forEach { range ->
        val matched = tokens.subList(range.first, range.last + 1).map { it.text }
        println("  [$range] → $matched")
    }

    println()
    println("=== matchesFully: exact vs partial ===")
    val exactText = "AIG-Life 99,00"
    val exactTokens = tokenizer.tokenize(text = exactText)
    println("'$exactText' fully matches: ${matcher.matchesFully(tokens = exactTokens, pattern = wordAmount)}")

    val partialText = "AIG-Life 99,00 extra"
    val partialTokens = tokenizer.tokenize(text = partialText)
    println("'$partialText' fully matches: ${matcher.matchesFully(tokens = partialTokens, pattern = wordAmount)}")

    println()
    println("=== findAll: no match ===")
    val noMatchText = "salary october payout"
    val noMatchRanges = matcher.findAll(text = noMatchText, pattern = wordAmount)
    println("'$noMatchText' match count: ${noMatchRanges.size}")

    println()
    println("=== pattern with optional WORD ===")
    val optionalPattern = TokenPattern.of {
        type(type = TokenTypeEnum.WORD)
        optional(type = TokenTypeEnum.WORD)
        type(type = TokenTypeEnum.AMOUNT)
    }
    val optText = "rent 500,00 monthly payment 200,00"
    val optTokens = tokenizer.tokenize(text = optText)
    val optRanges = matcher.findAll(tokens = optTokens, pattern = optionalPattern)
    println("text    : $optText")
    optRanges.forEach { range ->
        val matched = optTokens.subList(range.first, range.last + 1).map { it.text }
        println("  [$range] → $matched")
    }
}
