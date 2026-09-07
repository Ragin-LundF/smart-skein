package io.skein.classify.domain

/**
 * One train/holdout partition of a corpus, produced by
 * [io.skein.classify.application.StratifiedSplitter].
 *
 * Deliberately one type for both a single holdout split and each fold of a k-fold run — a separate
 * `Fold` type would be structurally identical.
 */
data class DatasetSplit(
    val training: List<LabeledFeatures>,
    val holdout: List<LabeledFeatures>,
)
