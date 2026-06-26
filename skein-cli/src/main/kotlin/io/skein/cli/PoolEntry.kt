package io.skein.cli

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Prediction

/**
 * One unlabeled row in the candidate pool. The [features] are cached after the first vectorization
 * so re-scoring across rounds never re-hashes the text (the dominant cost). [prediction] and
 * [uncertainty] hold the result of the most recent scoring; identity equality is used so the row can
 * be removed from the pool once labeled.
 */
class PoolEntry(val row: MutableMap<String, Any?>) {
    var features: FeatureVector? = null
    lateinit var prediction: Prediction
    var uncertainty: Double = 0.0
}
