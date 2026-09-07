package io.skein.classify.application

import io.skein.classify.domain.AttributionModeEnum
import io.skein.classify.domain.Calibration
import io.skein.classify.domain.ClassificationMetrics
import io.skein.classify.domain.Explanation
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
import io.skein.classify.infrastructure.NaiveBayesClassifier
import io.skein.classify.spi.Classifier
import io.skein.classify.spi.FeatureStore
import kotlin.random.Random

/** Contributions returned by [ClassificationService.explain] unless the caller asks for more. */
private const val DEFAULT_EXPLANATION_LIMIT = 10

/**
 * Orchestrates classification for one schema: maps records, vectorizes their feature text, and
 * drives a [Classifier] and a [FeatureStore]. One engine = one schema = one model.
 *
 * [schema], [classifier] and [featureStore] are exposed as read-only views so evaluation and
 * persistence can reach the model without this class growing those responsibilities. [mapper] and
 * the vectorizer stay private — they are the implementation of the record-to-features path rather
 * than collaborators the caller supplied.
 *
 * [privacyMode] is required (no default) and is a public, deliberate choice. In [FeatureStore]s
 * that only retain features (the in-memory default), both modes behave identically — the
 * [PrivacyModeEnum.ENCRYPTED_SOURCE] retention of original content is provided by a store that
 * supports encryption (see `skein-store-postgres`).
 */
class ClassificationService(
    val schema: Schema,
    val privacyMode: PrivacyModeEnum,
    hashingConfig: HashingConfig,
    val classifier: Classifier = NaiveBayesClassifier(),
    val featureStore: FeatureStore = InMemoryFeatureStore(),
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

    /**
     * Probability calibration applied to every [classify]. Defaults to [Calibration.NONE], so
     * predictions are bit-identical to an uncalibrated engine until this is set. Fit it with
     * [fitCalibration], or with [TemperatureCalibrator] directly, on data the model has **not**
     * learned from.
     *
     * Deliberately a mutable property rather than a constructor parameter, so the existing
     * constructor signature is untouched. Volatile because classification is lock-free and
     * concurrent, matching how the classifiers publish their snapshots.
     */
    @Volatile
    var calibration: Calibration = Calibration.NONE

    /** Predicts the label of a record, with [calibration] applied. */
    fun classify(record: Record): Prediction {
        val features = vectorizer.vectorize(text = mapper.map(record = record).featureText)
        return PredictionFactory.fromLogScores(
            logScores = classifier.logScores(features = features),
            calibration = calibration,
        )
    }

    /**
     * [classify], or `null` when the winning label's calibrated confidence falls below
     * [minConfidence] — the abstain path for callers that would rather return nothing than guess.
     */
    fun classifyOrNull(record: Record, minConfidence: Double): Prediction? {
        val prediction = classify(record = record)
        return if (prediction.isConfident(minConfidence = minConfidence)) prediction else null
    }

    /**
     * Why the model chose the label it did for [record]: the ranked per-feature contributions behind
     * the winning label, carrying the **calibrated** probability so an explanation always agrees
     * with [classify].
     *
     * Returns `null` when the configured classifier cannot attribute its score (see
     * [io.skein.classify.spi.Classifier.explain]).
     *
     * [mode] defaults to [AttributionModeEnum.BUCKETS_ONLY], which reveals no text.
     * [AttributionModeEnum.WITH_NGRAMS] additionally resolves each bucket to a representative
     * n-gram of this record — read that enum's note before using it.
     *
     * ponytail: costs one score lookup per label per non-zero feature, so it is fine interactively
     * or per reviewed row, and wrong inside a batch loop over millions of records. Upgrade path:
     * batch the per-label term lookups.
     */
    fun explain(
        record: Record,
        limit: Int = DEFAULT_EXPLANATION_LIMIT,
        mode: AttributionModeEnum = AttributionModeEnum.BUCKETS_ONLY,
    ): Explanation? {
        val featureText = mapper.map(record = record).featureText
        val features = vectorizer.vectorize(text = featureText)
        val prediction = PredictionFactory.fromLogScores(
            logScores = classifier.logScores(features = features),
            calibration = calibration,
        )
        val explanation = classifier.explain(features = features, label = prediction.label, limit = limit)
            ?: return null
        val calibrated = explanation.copy(probability = prediction.confidence)
        if (mode == AttributionModeEnum.BUCKETS_ONLY) {
            return calibrated
        }
        val ngrams = vectorizer.ngramsByBucket(text = featureText)
        return calibrated.copy(
            contributions = calibrated.contributions.map { contribution ->
                contribution.copy(ngram = ngrams[contribution.featureIndex])
            },
        )
    }

    /**
     * Fits a temperature on [heldOut] and installs it as [calibration], returning what was fitted.
     *
     * [heldOut] must be disjoint from what this engine learned from; see [TemperatureCalibrator].
     */
    fun fitCalibration(heldOut: List<LabeledFeatures>): Calibration {
        val fitted = TemperatureCalibrator().fit(heldOut = heldOut, classifier = classifier)
        calibration = fitted
        return fitted
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
