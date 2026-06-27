package io.skein.classify

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.infrastructure.NaiveBayesClassifier
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State

/** Throughput of [NaiveBayesClassifier.classify] (model scoring) on a pre-trained model. */
@State(Scope.Benchmark)
open class ScoringBenchmark {

    private val vectorizer = HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L))
    private val classifier = NaiveBayesClassifier()
    private val sample: FeatureVector = vectorizer.vectorize(text = "rent payment for the apartment")

    init {
        classifier.learn(
            features = vectorizer.vectorize(text = "rent transfer landlord monthly"),
            label = Label(value = "housing"),
        )
        classifier.learn(
            features = vectorizer.vectorize(text = "apartment rent standing order"),
            label = Label(value = "housing"),
        )
        classifier.learn(
            features = vectorizer.vectorize(text = "salary october payout employer"),
            label = Label(value = "income"),
        )
        classifier.learn(
            features = vectorizer.vectorize(text = "monthly salary payment"),
            label = Label(value = "income"),
        )
    }

    @Benchmark
    open fun classify(): Prediction {
        return classifier.classify(features = sample)
    }
}
