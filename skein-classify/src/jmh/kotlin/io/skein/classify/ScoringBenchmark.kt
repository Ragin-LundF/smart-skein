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

    private val vectorizer = HashingVectorizer(HashingConfig(key0 = 1L, key1 = 2L))
    private val classifier = NaiveBayesClassifier()
    private val sample: FeatureVector = vectorizer.vectorize("rent payment for the apartment")

    init {
        classifier.learn(vectorizer.vectorize("rent transfer landlord monthly"), Label("housing"))
        classifier.learn(vectorizer.vectorize("apartment rent standing order"), Label("housing"))
        classifier.learn(vectorizer.vectorize("salary october payout employer"), Label("income"))
        classifier.learn(vectorizer.vectorize("monthly salary payment"), Label("income"))
    }

    @Benchmark
    open fun classify(): Prediction {
        return classifier.classify(sample)
    }
}
