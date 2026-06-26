package io.skein.cli

import io.skein.classify.application.ClassificationService
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier
import io.skein.classify.spi.Classifier
import java.nio.file.Path

/**
 * Owns a [ClassificationService] together with the [InMemoryFeatureStore] it writes to, so the CLI
 * can export the learned observations and reload them later. Persistence is observation replay: the
 * model file holds the schema, hashing key and labeled feature vectors, and [restore] rebuilds the
 * classifier by replaying them via `ClassificationService.retrain`.
 *
 * Always [PrivacyModeEnum.FEATURES_ONLY] — only irreversible hashed features are kept, never source
 * content, which is what makes the exported model file safe to persist.
 */
class CliEngine private constructor(
    val service: ClassificationService,
    val classifier: ClassifierKindEnum,
    private val store: InMemoryFeatureStore,
    private val schema: Schema,
    private val hashingConfig: HashingConfig,
) {

    val labelColumn: String get() = schema.labelField.name

    /** Writes the current model (schema, hashing key, classifier kind, observations) to [path]. */
    fun save(path: Path) {
        ModelStore.save(
            path = path,
            schema = schema,
            classifier = classifier,
            hashingConfig = hashingConfig,
            observations = store.all(),
        )
    }

    companion object {

        /** A new untrained engine for [schema] using [hashingConfig] and the chosen [classifier]. */
        fun fresh(schema: Schema, classifier: ClassifierKindEnum, hashingConfig: HashingConfig): CliEngine {
            val store = InMemoryFeatureStore()
            val service = serviceFor(
                schema = schema,
                classifier = classifier,
                hashingConfig = hashingConfig,
                store = store,
            )
            return CliEngine(
                service = service,
                classifier = classifier,
                store = store,
                schema = schema,
                hashingConfig = hashingConfig,
            )
        }

        /** Rebuilds an engine from a [model] by replaying its observations ([epochs] passes for SGD). */
        fun restore(model: LoadedModel, epochs: Int): CliEngine {
            val store = InMemoryFeatureStore()
            store.addAll(observations = model.observations)
            val service = serviceFor(
                schema = model.schema,
                classifier = model.classifier,
                hashingConfig = model.hashingConfig,
                store = store,
            )
            service.retrain(epochs = epochs)
            return CliEngine(
                service = service,
                classifier = model.classifier,
                store = store,
                schema = model.schema,
                hashingConfig = model.hashingConfig,
            )
        }

        private fun serviceFor(
            schema: Schema,
            classifier: ClassifierKindEnum,
            hashingConfig: HashingConfig,
            store: InMemoryFeatureStore,
        ): ClassificationService {
            return ClassificationService(
                schema = schema,
                privacyMode = PrivacyModeEnum.FEATURES_ONLY,
                hashingConfig = hashingConfig,
                classifier = classifierFor(kind = classifier),
                featureStore = store,
            )
        }

        private fun classifierFor(kind: ClassifierKindEnum): Classifier {
            return when (kind) {
                ClassifierKindEnum.NAIVE_BAYES -> NaiveBayesClassifier()
                ClassifierKindEnum.LOGISTIC_REGRESSION -> LogisticRegressionSgdClassifier()
            }
        }
    }
}
