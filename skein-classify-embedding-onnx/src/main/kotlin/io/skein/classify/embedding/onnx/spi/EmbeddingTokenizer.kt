package io.skein.classify.embedding.onnx.spi

import io.skein.classify.embedding.onnx.domain.TokenizedText

/**
 * Port for turning text into the vocabulary ids a model consumes.
 *
 * Separate from [EmbeddingRuntime] because they are separate artifacts that must agree: a model is
 * trained against one specific vocabulary, and pairing it with another produces ids that mean
 * different words. Nothing detects that at runtime — the ids are all in range, the model runs, and
 * the embeddings are nonsense — so [identity] is folded into the vectorizer's fingerprint.
 *
 * Implementations hold native resources and must be closed.
 */
interface EmbeddingTokenizer : AutoCloseable {

    /**
     * Tokenizes [texts] into a rectangular batch, padding the shorter ones and marking the padding
     * in each [TokenizedText.attentionMask].
     */
    fun tokenize(texts: List<String>): List<TokenizedText>

    /**
     * A stable identifier for this vocabulary and its settings, for the model fingerprint. Two
     * tokenizers sharing an identity must produce identical ids for identical text.
     */
    fun identity(): String
}
