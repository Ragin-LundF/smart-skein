package io.skein.examples

import io.skein.classify.domain.Label
import io.skein.extract.domain.ExtractionResult

/** Outcome of the end-to-end pipeline: the predicted [category], its [confidence], and the
 *  structured fields [extracted] by the rules routed for that category. */
data class PipelineResult(
    val category: Label,
    val confidence: Double,
    val extracted: ExtractionResult,
)
