package io.skein.cli

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.ClassifierFactory
import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.application.HashingVectorizer
import io.skein.classify.application.LoadedModel
import io.skein.classify.application.ModelStore
import io.skein.classify.application.RecordMapper
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
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
 *
 * [vectorize] and [classify] split the two halves of `service.classify(record)` so a caller can
 * vectorize a record **once** and then re-score the cached [FeatureVector] cheaply as the model
 * changes — the basis of the scalable selection in [PoolSelector]. Both are thread-safe:
 * [HashingVectorizer] keeps per-thread scratch and the classifier scores a volatile snapshot, so the
 * same engine can vectorize/score many records concurrently (as long as nothing is learning).
 *
 * The schema, feature store and classifier are read back off [service] rather than duplicated here.
 */
class CliEngine private constructor(
    val service: ClassificationService,
    val classifier: ClassifierKindEnum,
    private val hashingConfig: HashingConfig,
) {

    private val vectorizer = HashingVectorizer(config = hashingConfig)
    private val mapper = RecordMapper(schema = service.schema)

    val labelColumn: String get() = service.schema.labelField.name

    /** Maps and hashes [record] into its sparse feature vector (the expensive, cacheable half). */
    fun vectorize(record: Record): FeatureVector {
        return vectorizer.vectorize(text = mapper.map(record = record).featureText)
    }

    /**
     * Scores an already-vectorized record against the current model (cheap, lock-free), applying the
     * engine's calibration so this agrees with `service.classify(record)`.
     */
    fun classify(features: FeatureVector): Prediction {
        return PredictionFactory.fromLogScores(
            logScores = service.classifier.logScores(features = features),
            calibration = service.calibration,
        )
    }

    /** True once the model has seen at least one labeled observation and can make predictions. */
    fun isTrained(): Boolean {
        return service.classifier.labels().isNotEmpty()
    }

    /** Writes the current model (schema, hashing key, classifier kind, observations) to [path]. */
    fun save(path: Path) {
        ModelStore.save(
            path = path,
            schema = service.schema,
            classifier = classifier,
            hashingConfig = hashingConfig,
            observations = service.featureStore.all(),
            calibration = service.calibration,
            hyperparameters = service.classifier.hyperparameters(),
        )
    }

    companion object {

        /** A new untrained engine for [schema] using [hashingConfig] and the chosen [classifier]. */
        fun fresh(schema: Schema, classifier: ClassifierKindEnum, hashingConfig: HashingConfig): CliEngine {
            return CliEngine(
                service = serviceFor(
                    schema = schema,
                    hashingConfig = hashingConfig,
                    model = ClassifierFactory.create(kind = classifier),
                    store = InMemoryFeatureStore(),
                ),
                classifier = classifier,
                hashingConfig = hashingConfig,
            )
        }

        /**
         * Rebuilds an engine from a [model] by replaying its observations ([epochs] passes for SGD).
         * Naive Bayes is an exact incremental algorithm: multiple epochs inflate feature counts via
         * Laplace-smoothing interaction and produce a different (incorrect) model. It is always
         * rebuilt with a single pass regardless of [epochs].
         */
        fun restore(model: LoadedModel, epochs: Int): CliEngine {
            val store = InMemoryFeatureStore()
            store.addAll(observations = model.observations)
            val service = serviceFor(
                schema = model.schema,
                hashingConfig = model.hashingConfig,
                model = ClassifierFactory.create(
                    kind = model.classifier,
                    hyperparameters = model.hyperparameters,
                ),
                store = store,
            )
            val effectiveEpochs = if (model.classifier == ClassifierKindEnum.NAIVE_BAYES) 1 else epochs
            service.retrain(epochs = effectiveEpochs)
            service.calibration = model.calibration
            return CliEngine(
                service = service,
                classifier = model.classifier,
                hashingConfig = model.hashingConfig,
            )
        }

        private fun serviceFor(
            schema: Schema,
            hashingConfig: HashingConfig,
            model: Classifier,
            store: InMemoryFeatureStore,
        ): ClassificationService {
            return ClassificationService(
                schema = schema,
                privacyMode = PrivacyModeEnum.FEATURES_ONLY,
                hashingConfig = hashingConfig,
                classifier = model,
                featureStore = store,
            )
        }
    }
}
