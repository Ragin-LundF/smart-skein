package io.skein.extract.domain

/** Character span of an extracted value in the source text: `[startOffset, endOffset)`. */
data class SourceSpan(val startOffset: Int, val endOffset: Int)
