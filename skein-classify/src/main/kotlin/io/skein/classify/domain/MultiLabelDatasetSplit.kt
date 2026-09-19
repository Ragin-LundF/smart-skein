package io.skein.classify.domain

/**
 * One train/holdout partition of a multi-label corpus. The multi-label sibling of [DatasetSplit],
 * and like it, one type serves both a single holdout and each fold of a k-fold run.
 */
data class MultiLabelDatasetSplit(
    val training: List<MultiLabeledFeatures>,
    val holdout: List<MultiLabeledFeatures>,
)
