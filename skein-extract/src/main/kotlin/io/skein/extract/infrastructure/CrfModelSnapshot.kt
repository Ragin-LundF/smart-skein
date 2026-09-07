package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag

/**
 * The complete restorable state of a [CrfSequenceLabeler]: its weights, its hyperparameters and how
 * far training has progressed.
 *
 * [tagOrder] is a list, not a set, because the order is semantic — Viterbi decoding and the
 * arg-max both break ties toward the lowest index, so reordering the tags silently changes what the
 * model predicts.
 */
data class CrfModelSnapshot(
    val tagOrder: List<Tag>,
    val startWeights: Map<Tag, Double>,
    val transitionWeights: Map<Pair<Tag, Tag>, Double>,
    val stateWeights: Map<Pair<Tag, String>, Double>,
    val featureCounts: Map<String, Int>,
    val initialLearningRate: Double,
    val decayRate: Double,
    val l2Regularization: Double,
    val step: Long,
)
