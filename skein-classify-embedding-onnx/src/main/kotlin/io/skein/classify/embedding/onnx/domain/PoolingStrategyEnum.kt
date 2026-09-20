package io.skein.classify.embedding.onnx.domain

/**
 * How a sequence of per-token vectors is collapsed into one vector for the whole text.
 *
 * A sentence encoder emits one vector per token; a classifier wants one per record. Which
 * reduction is correct is a property of **how the model was trained**, not a preference — using
 * the wrong one does not fail, it just produces systematically worse vectors. Check the model
 * card.
 */
enum class PoolingStrategyEnum {

    /**
     * The mean of the vectors of the real tokens, ignoring padding. The right default: it is what
     * sentence-transformers models are trained with, and nearly every model published for
     * similarity or retrieval expects it.
     */
    MEAN,

    /**
     * The first token's vector, which for a BERT-family model is the `[CLS]` position.
     *
     * Correct only for models whose training put a sentence representation there. On a model
     * trained for mean pooling this silently returns one token's vector as if it summarised the
     * sentence.
     */
    CLS,
}
