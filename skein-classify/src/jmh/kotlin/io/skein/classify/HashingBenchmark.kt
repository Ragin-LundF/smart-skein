package io.skein.classify

import io.skein.classify.infrastructure.SipHash
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State

/** Throughput of the keyed [SipHash] PRF on a short n-gram-sized input (the per-n-gram hot path). */
@State(Scope.Benchmark)
open class HashingBenchmark {

    private val data = "insurance-premium".toByteArray()
    private val key0 = 0x0706050403020100L
    private val key1 = 0x0f0e0d0c0b0a0908L

    @Benchmark
    open fun sipHash(): Long {
        return SipHash.hash(data, key0, key1)
    }
}
