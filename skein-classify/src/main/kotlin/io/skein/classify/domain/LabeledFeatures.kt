package io.skein.classify.domain

/** A feature vector paired with its observed label — the unit of training data. */
data class LabeledFeatures(val label: Label, val features: FeatureVector)
