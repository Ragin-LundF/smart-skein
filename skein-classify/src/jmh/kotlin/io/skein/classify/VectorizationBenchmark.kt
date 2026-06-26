package io.skein.classify

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State

/** Throughput of [HashingVectorizer.vectorize] (char + word n-gram hashing) on a typical record. */
@State(Scope.Benchmark)
open class VectorizationBenchmark {

    private val text = "rent transfer landlord monthly apartment payment standing order"
    private val vectorizer = HashingVectorizer(HashingConfig(key0 = 1L, key1 = 2L))

    @Benchmark
    open fun vectorize(): FeatureVector {
        return vectorizer.vectorize(text)
    }
}
