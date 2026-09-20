package io.skein.classify.embedding.onnx.infrastructure

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import io.skein.classify.embedding.onnx.domain.TokenizedText
import io.skein.classify.embedding.onnx.spi.EmbeddingTokenizer
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.io.path.readBytes

/**
 * Tokenizes with the Hugging Face tokenizer that shipped alongside the model.
 *
 * The vocabulary must be the model's own. Ids from another vocabulary are still in range and the
 * model still runs — it simply embeds different words than the text contained, with no error
 * anywhere. [identity] therefore digests the `tokenizer.json` itself, so the pairing is covered by
 * the vectorizer's fingerprint and a mismatch is refused at model load rather than discovered in
 * production.
 *
 * Padding is applied here rather than left to the tokenizer's own configuration. A batch has to be
 * rectangular, whether or not the model's `tokenizer.json` happens to enable padding, and doing it
 * explicitly means the attention mask is built from the same decision.
 */
class HuggingFaceEmbeddingTokenizer(
    tokenizerPath: Path,
    private val maxTokens: Int = DEFAULT_MAX_TOKENS,
) : EmbeddingTokenizer {

    init {
        require(value = maxTokens > 0) { "maxTokens must be positive, got $maxTokens" }
    }

    private val tokenizer: HuggingFaceTokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath)

    private val digest: String = digestOf(bytes = tokenizerPath.readBytes())

    override fun tokenize(texts: List<String>): List<TokenizedText> {
        require(value = texts.isNotEmpty()) { "cannot tokenize an empty batch" }
        val encoded = texts.map { text ->
            val encoding = tokenizer.encode(text)
            // Truncating here bounds the cost of one outlier record: attention is quadratic in
            // sequence length, so a single pathological input would otherwise dominate its batch.
            val length = minOf(a = encoding.ids.size, b = maxTokens)
            encoding.ids.copyOf(newSize = length) to encoding.attentionMask.copyOf(newSize = length)
        }
        val width = encoded.maxOf { (ids, _) -> ids.size }
        return encoded.map { (ids, mask) ->
            TokenizedText(
                ids = ids.copyOf(newSize = width),
                // copyOf pads with zeros, which is exactly "not a real token" in an attention mask.
                attentionMask = mask.copyOf(newSize = width),
            )
        }
    }

    override fun identity(): String {
        return "huggingface:$digest:maxTokens=$maxTokens"
    }

    override fun close() {
        tokenizer.close()
    }

    private fun digestOf(bytes: ByteArray): String {
        return MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    private companion object {
        /** Matches the 512-position limit of the BERT-family encoders this is usually paired with. */
        const val DEFAULT_MAX_TOKENS = 512
    }
}
