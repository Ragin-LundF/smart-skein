package io.skein.text

import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State

/** Throughput of [TypedTokenizer.tokenize] on a representative mixed-content line. */
@State(Scope.Benchmark)
open class TokenizationBenchmark {

    private val line = "booked 2024-12-31 amount 1234.56 AIG-Life CustomerNumber AB12345 insurance premium"
    private val tokenizer = TypedTokenizer()

    @Benchmark
    open fun tokenize(): List<Token> {
        return tokenizer.tokenize(text = line)
    }
}
