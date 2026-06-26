package io.skein.extract.domain

/**
 * One value pulled from the source text.
 *
 * @property name slot name (component name for repeating groups)
 * @property value the real extracted text — extraction returns values, it does not hash or destroy them
 * @property span where the value sits in the source
 * @property confidence `[0, 1]`; deterministic pattern matches are `1.0`
 * @property groupIndex occurrence index for repeating-group slots so paired fields can be re-associated;
 *   `null` for single-value slots
 */
data class ExtractedField(
    val name: String,
    val value: String,
    val span: SourceSpan,
    val confidence: Double = 1.0,
    val groupIndex: Int? = null,
)
