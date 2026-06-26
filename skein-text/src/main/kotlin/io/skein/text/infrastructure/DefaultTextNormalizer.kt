package io.skein.text.infrastructure

import io.skein.text.spi.TextNormalizer

/**
 * Default normalizer: removes control characters, collapses any run of whitespace
 * into a single space, trims, and lowercases.
 *
 * Deliberately conservative — it only canonicalizes spacing and case so that
 * downstream tokenization and broken-word repair see stable, comparable input.
 */
class DefaultTextNormalizer : TextNormalizer {

    override fun normalize(raw: String): String {
        val withoutControls = CONTROL_CHARS.replace(input = raw, replacement = " ")
        val collapsed = WHITESPACE.replace(input = withoutControls, replacement = " ")
        return collapsed.trim().lowercase()
    }

    private companion object {
        val WHITESPACE = Regex(pattern = "\\s+")
        val CONTROL_CHARS = Regex(pattern = "\\p{Cntrl}")
    }
}
