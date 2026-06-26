package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
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
    private var snapshot = NaiveBayesSnapshot.empty(smoothingAlpha)
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

    override fun classify(features: FeatureVector): Prediction {
        val current = snapshot
        check(current.isTrained()) { "classifier has not been trained" }
        return current.predict(features)
    }

    override fun labels(): Set<Label> {
        return snapshot.labels().toSet()
    }

    override fun forget() {
        synchronized(writeLock) {
            seenFeatures.clear()
            snapshot = NaiveBayesSnapshot.empty(smoothingAlpha)
        }
    }

    private companion object {
        const val DEFAULT_SMOOTHING_ALPHA = 1.0
    }
}
