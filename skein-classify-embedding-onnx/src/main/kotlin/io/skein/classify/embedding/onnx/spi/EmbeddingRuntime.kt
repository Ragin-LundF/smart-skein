package io.skein.classify.embedding.onnx.spi

import io.skein.classify.embedding.onnx.domain.TokenEmbeddings
import io.skein.classify.embedding.onnx.domain.TokenizedText

/**
 * Port for the thing that actually runs an embedding model.
 *
 * This is the only place in the module that has to know about a model format or an inference
 * engine, which is deliberate: everything interesting around it — pooling, normalisation, caching,
 * the fingerprint, the conversion to a `FeatureVector` — is arithmetic and bookkeeping that would
 * otherwise be untestable without a real model on disk.
 *
 * Implementations hold native resources and must be closed.
 */
interface EmbeddingRuntime : AutoCloseable {

    /** Width of one token's vector, and therefore of the pooled embedding. */
    fun hiddenSize(): Int

    /**
     * Runs the model over a whole [batch] at once.
     *
     * A batch rather than a single text because that is where the performance is. A transformer
     * encoder costs roughly 1–3 ms per record on one core, so embedding a million records one at a
     * time is over half an hour of pure inference per training run; batching amortises the fixed
     * per-call overhead and lets the runtime use its own parallelism.
     *
     * The returned list is aligned with [batch] by position.
     */
    fun encode(batch: List<TokenizedText>): List<TokenEmbeddings>
}
