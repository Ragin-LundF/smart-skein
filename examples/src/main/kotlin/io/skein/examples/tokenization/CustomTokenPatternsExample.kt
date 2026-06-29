package io.skein.examples.tokenization

import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.TokenPatternConfig
import io.skein.text.domain.TokenTypeEnum

fun runCustomTokenPatternsExample() {
    val orderCodeRule = Regex(pattern = "ORD-\\d+") to TokenTypeEnum.ALPHANUMERIC

    val domainConfig = TokenPatternConfig(
        typedRules = listOf(orderCodeRule) + TokenPatternConfig.GERMAN.typedRules,
    )

    val defaultTokenizer = TypedTokenizer()
    val domainTokenizer = TypedTokenizer(patterns = domainConfig)

    val text = "ORD-4711 shipped 12,50 on 01.06.2024"

    val defaultTokens = defaultTokenizer.tokenize(text = text)
    val domainTokens = domainTokenizer.tokenize(text = text)

    println("=== Input ===")
    println("  $text")
    println()

    println("=== GERMAN default config (${defaultTokens.size} tokens) ===")
    defaultTokens.forEach { token ->
        println("  '${token.text}'  → ${token.type}")
    }

    println()
    println("=== Domain config with ORD-\\d+ rule prepended (${domainTokens.size} tokens) ===")
    domainTokens.forEach { token ->
        println("  '${token.text}'  → ${token.type}")
    }

    println()
    println("=== Diff ===")
    val defaultOrd = defaultTokens.first { it.text == "ORD-4711" }.type
    val domainOrd = domainTokens.first { it.text == "ORD-4711" }.type
    println("  'ORD-4711' with GERMAN : $defaultOrd")
    println("  'ORD-4711' with domain : $domainOrd")
    println("  DATE and AMOUNT classification is unchanged in both configs.")
}
