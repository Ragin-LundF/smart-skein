package io.skein.cli

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.UncertaintyStrategyEnum

internal val LABEL_FLAGS = setOf(
    "input", "label-col", "out", "model", "classifier",
    "budget", "batch", "strategy", "epochs", "key", "scan-limit", "delimiter",
)
internal val PREDICT_FLAGS = setOf("input", "model", "out", "epochs", "delimiter")
internal val EXPORT_FLAGS = setOf("model", "out")

/** Parses `--name value` tokens into a map. Rejects a token that is not a `--flag` or has no value. */
internal fun parseFlags(tokens: List<String>): Map<String, String> {
    val flags = LinkedHashMap<String, String>()
    var index = 0
    while (index < tokens.size) {
        val token = tokens[index]
        require(value = token.startsWith(prefix = "--")) { "expected a --flag, got '$token'" }
        val name = token.removePrefix(prefix = "--")
        require(value = index + 1 < tokens.size) { "missing value for --$name" }
        flags[name] = tokens[index + 1]
        index += 2
    }
    return flags
}

internal fun Map<String, String>.required(name: String): String {
    return this[name] ?: throw IllegalArgumentException("missing required --$name")
}

/** Fails on any flag not in [allowed], so a typo like `--budgett` is caught instead of silently ignored. */
internal fun requireKnownFlags(flags: Map<String, String>, allowed: Set<String>) {
    val unknown = flags.keys - allowed
    require(value = unknown.isEmpty()) {
        "unknown flag(s): ${unknown.joinToString(separator = ", ") { name -> "--$name" }}"
    }
}

internal fun parseClassifier(value: String?): ClassifierKindEnum {
    return when (value) {
        null, "nb" -> ClassifierKindEnum.NAIVE_BAYES
        "logreg" -> ClassifierKindEnum.LOGISTIC_REGRESSION
        else -> throw IllegalArgumentException("unknown --classifier '$value' (use nb or logreg)")
    }
}

internal fun parseStrategy(value: String?): UncertaintyStrategyEnum {
    return when (value) {
        null, "margin" -> UncertaintyStrategyEnum.MARGIN
        "least-confidence" -> UncertaintyStrategyEnum.LEAST_CONFIDENCE
        "entropy" -> UncertaintyStrategyEnum.ENTROPY
        else -> throw IllegalArgumentException("unknown --strategy '$value'")
    }
}

internal fun parseDelimiter(value: String?): Char {
    return when (value) {
        null -> ','
        "\\t" -> '\t'
        else -> {
            require(value = value.length == 1) { "--delimiter must be a single character (or \\\\t for tab), got '$value'" }
            value[0]
        }
    }
}

internal fun parseKey(value: String?): HashingConfig? {
    if (value == null) {
        return null
    }
    val parts = value.split(",")
    require(value = parts.size == 2) { "--key must be '<key0>,<key1>'" }
    return HashingConfig(key0 = parts[0].trim().toLong(), key1 = parts[1].trim().toLong())
}
