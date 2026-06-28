package io.skein.classify.application

import io.skein.classify.domain.ClassificationMetrics
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
import io.skein.classify.infrastructure.NaiveBayesClassifier
import io.skein.classify.spi.Classifier
import io.skein.classify.spi.FeatureStore
import kotlin.random.Random

/**
 * Orchestrates classification for one schema: maps records, vectorizes their feature text, and
 * drives a [Classifier] and a [FeatureStore]. One engine = one schema = one model.
 *
 * [privacyMode] is required (no default) and is a public, deliberate choice. In [FeatureStore]s
 * that only retain features (the in-memory default), both modes behave identically — the
 * [PrivacyModeEnum.ENCRYPTED_SOURCE] retention of original content is provided by a store that
 * supports encryption (see `skein-store-postgres`).
 */
class ClassificationService(
    schema: Schema,
    val privacyMode: PrivacyModeEnum,
    hashingConfig: HashingConfig,
    private val classifier: Classifier = NaiveBayesClassifier(),
    private val featureStore: FeatureStore = InMemoryFeatureStore(),
    private val mapper: RecordMapper = RecordMapper(schema),
) {

    private val vectorizer = HashingVectorizer(config = hashingConfig)

    /** Learns from a record whose label field carries the target label. */
    fun learn(record: Record) {
        val mapped = mapper.map(record = record)
        val label = mapped.label
            ?: throw IllegalArgumentException("cannot learn from a record without a label value")
        learnLabeled(featureText = mapped.featureText, label = label)
    }

    /** Learns from many records. Publishes one classifier snapshot instead of one per observation. */
    fun learnAll(records: Iterable<Record>) {
        val observations = records.map { record ->
            val mapped = mapper.map(record = record)
            val label = mapped.label
                ?: throw IllegalArgumentException("cannot learn from a record without a label value")
            val features = vectorizer.vectorize(text = mapped.featureText)
            val obs = LabeledFeatures(label = label, features = features)
            featureStore.add(obs)
            obs
        }
        classifier.learnAll(observations = observations)
    }

    /** Corrects a prediction by learning the record under the [correctLabel]. */
    fun feedback(record: Record, correctLabel: Label) {
        learnLabeled(featureText = mapper.map(record = record).featureText, label = correctLabel)
    }

    /** Predicts the label of a record. */
    fun classify(record: Record): Prediction {
        val features = vectorizer.vectorize(text = mapper.map(record = record).featureText)
        return classifier.classify(features = features)
    }

    /** Reports total observations and per-label counts learned so far. */
    fun metrics(): ClassificationMetrics {
        val perLabel = featureStore.all().groupingBy { observation -> observation.label }.eachCount()
        return ClassificationMetrics(totalObservations = featureStore.size(), perLabelCounts = perLabel)
    }

    /**
     * Rebuilds the model from scratch by replaying every stored observation [epochs] times.
     * Useful for classifiers that benefit from multiple passes (e.g. logistic regression). The
     * stored observations are kept; only the classifier state is reset.
     *
     * By default observations replay in stored order, so a retrain is deterministic and resumable.
     * Pass a [seed] to shuffle each epoch with that seed — still deterministic, but breaking the
     * stored order, which helps SGD convergence by decorrelating consecutive updates.
     */
    fun retrain(epochs: Int = 1, seed: Long? = null) {
        require(value = epochs >= 1) { "epochs must be at least 1" }
        classifier.forget()
        val observations = featureStore.all()
        repeat(times = epochs) { epoch ->
            val ordered = if (seed == null) {
                observations
            } else {
                observations.shuffled(random = Random(seed = seed + epoch))
            }
            classifier.learnAll(observations = ordered)
        }
    }

    /** Discards all learned state (model and stored observations). */
    fun forget() {
        classifier.forget()
        featureStore.clear()
    }

    private fun learnLabeled(featureText: String, label: Label) {
        val features = vectorizer.vectorize(text = featureText)
        classifier.learn(features = features, label = label)
        featureStore.add(LabeledFeatures(label = label, features = features))
    }
}
