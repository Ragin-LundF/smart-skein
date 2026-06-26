package io.skein.cli

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Schema

/**
 * A model restored from disk: everything needed to rebuild an engine and keep training. The
 * [observations] are replayed into a fresh classifier (see `ClassificationService.retrain`), so the
 * model state is reproduced exactly rather than stored as opaque classifier internals.
 */
data class LoadedModel(
    val schema: Schema,
    val classifier: ClassifierKindEnum,
    val hashingConfig: HashingConfig,
    val observations: List<LabeledFeatures>,
)
