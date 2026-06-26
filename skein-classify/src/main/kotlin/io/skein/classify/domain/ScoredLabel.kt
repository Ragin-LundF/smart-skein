package io.skein.classify.domain

/** A label paired with its predicted probability in `[0, 1]`. */
data class ScoredLabel(val label: Label, val probability: Double)
