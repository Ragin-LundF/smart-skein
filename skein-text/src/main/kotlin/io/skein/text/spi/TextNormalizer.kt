package io.skein.text.spi

/**
 * Port that turns a raw, possibly noisy string into a clean canonical form before
 * tokenization. Implementations live in the infrastructure layer so the domain
 * stays free of concrete normalization choices.
 */
interface TextNormalizer {

    /** Returns the canonical form of [raw]. Must be idempotent: `normalize(normalize(x)) == normalize(x)`. */
    fun normalize(raw: String): String
}
