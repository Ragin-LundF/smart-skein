package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory
import io.skein.classify.spi.Classifier
import kotlin.jvm.Volatile

/**
 * Incremental multinomial logistic regression trained with online stochastic gradient descent.
 *
 * Each label owns a sparse weight vector (only touched feature indices are stored) and a bias.
 * `learn` performs one SGD step per observation: it computes the softmax over the current weights
 * and nudges every label's weights by `learningRate * (probability - indicator) * featureValue`,
 * with optional L2 shrinkage. The step size follows a `1 / (1 + decayRate * step)` schedule
 * ([decayRate] = 0 keeps it constant). New labels are discovered on first sight and start from zero.
 *
 * Prediction is **lock-free**: trained state lives in an immutable [LogisticRegressionSnapshot] held
 * in a `@Volatile` reference. `learn` is serialized — it updates the mutable training maps, then
 * publishes a fresh snapshot — so a concurrent `classify` always reads a consistent snapshot.
 *
 * Compared with [NaiveBayesClassifier] this is the discriminative "v2": it usually needs several
 * passes over the data (see `ClassificationService.retrain`) but models correlated features better.
 */
class LogisticRegressionSgdClassifier(
    private val initialLearningRate: Double = DEFAULT_LEARNING_RATE,
    private val decayRate: Double = DEFAULT_DECAY_RATE,
    private val l2Regularization: Double = DEFAULT_L2_REGULARIZATION,
) : Classifier {

    private val weightsByLabel = HashMap<Label, HashMap<Int, Double>>()
    private val biasByLabel = HashMap<Label, Double>()
    private val writeLock = Any()
    private var step = 0L

    @Volatile
    private var snapshot = LogisticRegressionSnapshot.empty()

    override fun learnAll(observations: List<LabeledFeatures>) {
        synchronized(writeLock) {
            for (obs in observations) {
                ensureLabel(obs.label)
                val lr = initialLearningRate / (1.0 + decayRate * step)
                val probs = probabilities(features = obs.features)
                for (candidate in weightsByLabel.keys) {
                    val indicator = if (candidate == obs.label) 1.0 else 0.0
                    applyGradient(
                        label = candidate,
                        features = obs.features,
                        error = probs.getValue(key = candidate) - indicator,
                        learningRate = lr,
                    )
                }
                step++
            }
            snapshot = LogisticRegressionSnapshot.of(weightsByLabel = weightsByLabel, biasByLabel = biasByLabel)
        }
    }

    override fun learn(features: FeatureVector, label: Label) {
        synchronized(writeLock) {
            ensureLabel(label)
            val learningRate = initialLearningRate / (1.0 + decayRate * step)
            val probabilities = probabilities(features = features)
            for (candidate in weightsByLabel.keys) {
                val indicator = if (candidate == label) 1.0 else 0.0
                applyGradient(
                    label = candidate,
                    features = features,
                    error = probabilities.getValue(key = candidate) - indicator,
                    learningRate = learningRate,
                )
            }
            step++
            snapshot = LogisticRegressionSnapshot.of(weightsByLabel = weightsByLabel, biasByLabel = biasByLabel)
        }
    }

    override fun classify(features: FeatureVector): Prediction {
        val current = snapshot
        check(current.isTrained()) { "classifier has not been trained" }
        return current.predict(features = features)
    }

    override fun labels(): Set<Label> {
        return snapshot.labels()
    }

    override fun forget() {
        synchronized(writeLock) {
            weightsByLabel.clear()
            biasByLabel.clear()
            step = 0L
            snapshot = LogisticRegressionSnapshot.empty()
        }
    }

    private fun ensureLabel(label: Label) {
        weightsByLabel.getOrPut(key = label) { HashMap() }
        biasByLabel.putIfAbsent(label, 0.0)
    }

    private fun probabilities(features: FeatureVector): Map<Label, Double> {
        val logits = HashMap<Label, Double>()
        for (label in weightsByLabel.keys) {
            logits[label] = scoreFor(label = label, features = features)
        }
        return PredictionFactory.fromLogScores(logScores = logits)
            .alternatives
            .associate { scored -> scored.label to scored.probability }
    }

    private fun scoreFor(label: Label, features: FeatureVector): Double {
        val weights = weightsByLabel.getValue(key = label)
        var score = biasByLabel[label] ?: 0.0
        for (position in features.indices.indices) {
            score += (weights[features.indices[position]] ?: 0.0) * features.values[position].toDouble()
        }
        return score
    }

    private fun applyGradient(label: Label, features: FeatureVector, error: Double, learningRate: Double) {
        val weights = weightsByLabel.getValue(key = label)
        for (position in features.indices.indices) {
            val index = features.indices[position]
            val current = weights[index] ?: 0.0
            val gradient = error * features.values[position].toDouble() + l2Regularization * current
            weights[index] = current - learningRate * gradient
        }
        biasByLabel[label] = (biasByLabel[label] ?: 0.0) - learningRate * error
    }

    private companion object {
        const val DEFAULT_LEARNING_RATE = 0.1
        const val DEFAULT_DECAY_RATE = 0.0
        const val DEFAULT_L2_REGULARIZATION = 0.0
    }
}
