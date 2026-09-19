package io.skein.classify.domain

/**
 * One evaluated case: the label set that was true, and what the model predicted for it.
 *
 * The multi-label sibling of [PredictionOutcome]. [expected] may be empty — an explicit negative is
 * scored, and a model that invents a label for it takes a false positive, which is the point.
 */
data class MultiLabelOutcome(val expected: Set<Label>, val prediction: MultiLabelPrediction)
