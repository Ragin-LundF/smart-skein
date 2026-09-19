package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.LabelThresholds
import io.skein.classify.domain.Schema
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier

/**
 * A multi-label model read back from a `.skein` file.
 *
 * Unlike [LoadedModel], which returns the observations a single-label classifier replays, this
 * carries an **already-fitted** [classifier]: a batch-trained model has nothing to replay, and
 * re-fitting on load would turn opening a file into a training run.
 *
 * [hashingConfig] is `null` when the model was trained with a vectorizer that is not this library's
 * feature hashing — an external embedding model, for instance, for which those settings mean
 * nothing.
 */
data class LoadedMultiLabelModel(
    val schema: Schema,
    val classifier: MultiLabelLogisticClassifier,
    val fingerprint: VectorizerFingerprint,
    val thresholds: LabelThresholds,
    val hashingConfig: HashingConfig?,
)
