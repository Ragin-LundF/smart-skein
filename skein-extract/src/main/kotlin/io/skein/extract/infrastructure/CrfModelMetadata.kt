package io.skein.extract.infrastructure

import java.time.Instant

/**
 * The header of a saved CRF model, readable without inflating the weights — so an operator can ask
 * what retention policy a file was written under, by which version, and when.
 */
data class CrfModelMetadata(
    val formatMajor: Int,
    val formatMinor: Int,
    val createdAt: Instant,
    val writerVersion: String,
    val retention: FeatureRetentionEnum,
    val minLexicalOccurrences: Int,
)
