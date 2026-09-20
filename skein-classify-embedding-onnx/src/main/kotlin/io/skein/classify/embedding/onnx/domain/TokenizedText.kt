package io.skein.classify.embedding.onnx.domain

/**
 * One text after tokenization: the vocabulary ids the model consumes, and the mask saying which of
 * them are real.
 *
 * Padding exists because a batch has to be rectangular — every row of a tensor is the same length —
 * while the texts in it are not. [attentionMask] carries `1` for a real token and `0` for padding,
 * and honouring it is what stops a short text's vector from being dragged towards whatever the
 * padding token embeds to.
 *
 * A plain class rather than a `data class`: both fields are arrays, and a generated `equals` over an
 * array compares identity, which is a trap in something that looks like a value.
 */
class TokenizedText(val ids: LongArray, val attentionMask: LongArray) {

    init {
        require(value = ids.size == attentionMask.size) {
            "ids (${ids.size}) and attentionMask (${attentionMask.size}) must have equal length"
        }
        require(value = attentionMask.all { flag -> flag == 0L || flag == 1L }) {
            "attentionMask must hold only 0 or 1"
        }
    }

    /** Tokens that are not padding. */
    fun realTokenCount(): Int {
        return attentionMask.count { flag -> flag == 1L }
    }
}
