package io.skein.extract.domain

import io.skein.text.domain.PatternSignature

/** A group of source texts that share the same [signature] — a discovered recurring layout. */
data class TemplateCluster(
    val signature: PatternSignature,
    val members: List<String>,
)
