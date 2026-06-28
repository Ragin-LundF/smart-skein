package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Prediction
import io.skein.classify.spi.Classifier
import kotlin.jvm.Volatile

/**
 * Incremental multinomial Naive Bayes over sparse hashed features.
 *
 * Per label it accumulates the document count (prior) and the summed feature mass (likelihood),
 * applying additive (Laplace) smoothing with [smoothingAlpha] over the observed feature vocabulary.
 * Scoring reads calibrated probabilities via a numerically stable softmax.
 *
 * Prediction is **lock-free**: the trained state lives in an immutable [NaiveBayesSnapshot] held in
 * a `@Volatile` reference, so `classify` reads it without synchronization. `learn` is serialized and
 * publishes a new snapshot (copy-on-write), so a `classify` running concurrently with training sees
 * a consistent prior-or-next snapshot, never a torn one.
 */
class NaiveBayesClassifier(private val smoothingAlpha: Double = DEFAULT_SMOOTHING_ALPHA) : Classifier {

    @Volatile
    private var snapshot = NaiveBayesSnapshot.empty(smoothingAlpha = smoothingAlpha)
    private val seenFeatures = HashSet<Int>()
    private val writeLock = Any()

    override fun learn(features: FeatureVector, label: Label) {
        synchronized(writeLock) {
            for (position in features.indices.indices) {
                seenFeatures.add(features.indices[position])
            }
            snapshot = snapshot.withObservation(
                features = features,
                label = label,
                newVocabularySize = seenFeatures.size,
            )
        }
    }

    /**
     * Batch variant: accumulates counts in mutable maps and publishes exactly ONE snapshot at the
     * end instead of one per observation. Avoids the O(obs × map_copy) cost of the copy-on-write
     * protocol — critical for bulk restore of large models.
     */
    override fun learnAll(observations: List<LabeledFeatures>) {
        synchronized(writeLock) {
            val docCounts = HashMap<Label, Long>()
            val featSums = HashMap<Label, HashMap<Int, Double>>()
            val featMass = HashMap<Label, Double>()
            for (obs in observations) {
                docCounts[obs.label] = (docCounts[obs.label] ?: 0L) + 1L
                val sums = featSums.getOrPut(obs.label) { HashMap() }
                var mass = featMass[obs.label] ?: 0.0
                for (p in obs.features.indices.indices) {
                    val idx = obs.features.indices[p]
                    val v = obs.features.values[p].toDouble()
                    sums[idx] = (sums[idx] ?: 0.0) + v
                    mass += v
                    seenFeatures.add(idx)
                }
                featMass[obs.label] = mass
            }
            snapshot = NaiveBayesSnapshot(
                labelDocumentCounts = docCounts,
                featureSumsByLabel = featSums,
                featureMassByLabel = featMass,
                totalDocuments = observations.size.toLong(),
                vocabularySize = seenFeatures.size,
                smoothingAlpha = smoothingAlpha,
            )
        }
    }

    override fun classify(features: FeatureVector): Prediction {
        val current = snapshot
        check(current.isTrained()) { "classifier has not been trained" }
        return current.predict(features = features)
    }

    override fun labels(): Set<Label> {
        return snapshot.labels().toSet()
    }

    override fun forget() {
        synchronized(writeLock) {
            seenFeatures.clear()
            snapshot = NaiveBayesSnapshot.empty(smoothingAlpha = smoothingAlpha)
        }
    }

    private companion object {
        const val DEFAULT_SMOOTHING_ALPHA = 1.0
    }
}
