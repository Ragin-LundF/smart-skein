package io.skein.text.domain

/**
 * A single typed token produced by the tokenizer.
 *
 * @property text the exact source substring
 * @property type its structural classification
 * @property startOffset inclusive start index in the source string
 * @property endOffset exclusive end index in the source string
 */
data class Token(
    val text: String,
    val type: TokenTypeEnum,
    val startOffset: Int,
    val endOffset: Int,
)
