package io.skein.classify.embedding.onnx.domain

/**
 * Whether a pooled embedding is scaled to unit length.
 *
 * A deliberate enum rather than a boolean flag: `normalize = true` at a call site says nothing
 * about what changes, and the choice has to match the model's training to be meaningful.
 */
enum class NormalizationEnum {

    /**
     * Scale to unit L2 length. The usual choice for models trained with a cosine objective — which
     * is most retrieval and similarity models — because it makes a dot product a cosine similarity
     * and stops long inputs from carrying more weight than short ones purely through magnitude.
     */
    L2,

    /** Leave the pooled vector as the model produced it. */
    NONE,
}
