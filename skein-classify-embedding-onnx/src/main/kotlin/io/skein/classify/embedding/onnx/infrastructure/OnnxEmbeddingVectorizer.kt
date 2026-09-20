package io.skein.classify.embedding.onnx.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.embedding.onnx.application.EmbeddingCache
import io.skein.classify.embedding.onnx.domain.DenseVectors
import io.skein.classify.embedding.onnx.domain.NormalizationEnum
import io.skein.classify.embedding.onnx.domain.PoolingStrategyEnum
import io.skein.classify.embedding.onnx.spi.EmbeddingRuntime
import io.skein.classify.embedding.onnx.spi.EmbeddingTokenizer
import io.skein.classify.spi.BatchVectorizer
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.io.path.inputStream

/** Identifies this implementation in a [VectorizerFingerprint]. */
private const val KIND = "onnx-embedding"

/** Bytes read per digest update when hashing a model file. */
private const val DIGEST_BUFFER_BYTES = 1 shl 16

/**
 * A [io.skein.classify.spi.Vectorizer] backed by an external sentence-embedding model run through
 * ONNX Runtime.
 *
 * **What this buys.** Hashed n-grams match literal text: a model trained on "eggplant" learns
 * nothing about "aubergine". An embedding places related wording nearby in vector space, so the
 * classifier generalises past the exact strings it was trained on — which is precisely where a
 * rule-distilled model otherwise fails.
 *
 * **What it costs, and this is the part to decide on before adopting it.** Hashed n-grams run at
 * about 41 µs per record; a MiniLM-class transformer at int8 costs 1–3 ms per core, which is
 * 25–75x slower and enough to put a 1,000 records/s target at risk *on featurisation alone*.
 * Static embeddings (Model2Vec-class) sit in between at roughly 100–200 µs and keep most of the
 * semantic benefit. Batch with [vectorizeAll], give it a [cache], and measure before committing.
 *
 * The classifier itself gets **smaller**, not larger: 384 dimensions x 237 labels x 4 bytes is
 * 364 KB against 17 MB for a sparse n-gram model, and pruning stops being worth doing. The artifact
 * that matters becomes the embedding model, at 20–400 MB.
 *
 * Holds native resources — close it, which closes the runtime and tokenizer it was given.
 */
class OnnxEmbeddingVectorizer(
    private val runtime: EmbeddingRuntime,
    private val tokenizer: EmbeddingTokenizer,
    private val modelDigest: String,
    private val pooling: PoolingStrategyEnum = PoolingStrategyEnum.MEAN,
    private val normalization: NormalizationEnum = NormalizationEnum.L2,
    private val cache: EmbeddingCache? = null,
) : BatchVectorizer, AutoCloseable {

    private val fingerprint: VectorizerFingerprint = buildFingerprint()

    /**
     * Embeds one text.
     *
     * Present because [io.skein.classify.spi.Vectorizer] requires it, and it is the slow path: it
     * runs the model on a batch of one, paying the full per-call overhead for a single record.
     * Prefer [vectorizeAll] whenever more than one record is in hand — which, during training, is
     * always.
     */
    override fun vectorize(text: String): FeatureVector {
        return vectorizeAll(texts = listOf(text)).single()
    }

    /**
     * Embeds [texts] in one pass, returning vectors aligned with the input by position.
     *
     * Only the texts missing from the [cache] reach the model, and they go as a single batch. A
     * corpus with repeated records — which, after masking, most real corpora are — therefore costs
     * one forward pass per *distinct* record rather than per record.
     */
    override fun vectorizeAll(texts: List<String>): List<FeatureVector> {
        if (texts.isEmpty()) {
            return emptyList()
        }
        val pooled = arrayOfNulls<FloatArray>(texts.size)
        val missingPositions = ArrayList<Int>(texts.size)
        val missingTexts = ArrayList<String>(texts.size)

        texts.forEachIndexed { position, text ->
            val cached = cache?.get(key = cacheKey(text = text))
            if (cached == null) {
                missingPositions.add(element = position)
                missingTexts.add(element = text)
            } else {
                pooled[position] = cached
            }
        }

        if (missingTexts.isNotEmpty()) {
            embed(texts = missingTexts).forEachIndexed { index, embedding ->
                val position = missingPositions[index]
                pooled[position] = embedding
                cache?.put(key = cacheKey(text = missingTexts[index]), embedding = embedding)
            }
        }

        return pooled.map { embedding -> DenseVectors.asFeatureVector(vector = embedding!!) }
    }

    override fun dimension(): Int {
        return runtime.hiddenSize()
    }

    override fun fingerprint(): VectorizerFingerprint {
        return fingerprint
    }

    override fun close() {
        // Close both even if the first throws, so one failing native handle cannot leak the other.
        runtime.use { tokenizer.close() }
    }

    private fun embed(texts: List<String>): List<FloatArray> {
        return runtime.encode(batch = tokenizer.tokenize(texts = texts)).map { sequence ->
            val vector = sequence.pool(strategy = pooling)
            when (normalization) {
                NormalizationEnum.L2 -> DenseVectors.l2Normalize(vector = vector)
                NormalizationEnum.NONE -> vector
            }
        }
    }

    /**
     * Scopes a cached vector to the exact configuration that produced it, so a changed model,
     * tokenizer, pooling strategy or normalisation cannot be served a stale embedding.
     */
    private fun cacheKey(text: String): String {
        return "${fingerprint.configDigest}:${digestOf(text = text)}"
    }

    private fun buildFingerprint(): VectorizerFingerprint {
        val material = listOf(
            "model=$modelDigest",
            "tokenizer=${tokenizer.identity()}",
            "pooling=${pooling.name}",
            "normalization=${normalization.name}",
            "hiddenSize=${runtime.hiddenSize()}",
        ).joinToString(separator = "|")
        return VectorizerFingerprint(
            kind = KIND,
            dimension = runtime.hiddenSize(),
            configDigest = digestOf(text = material),
        )
    }

    private fun digestOf(text: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8))
        return digest.joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    companion object {

        /**
         * The SHA-256 of a model file's **contents**.
         *
         * Contents, never the filename or a version string: a model swapped in place under an
         * unchanged name is exactly the failure the fingerprint exists to catch, and a name-based
         * identity would miss it completely. Streamed rather than read whole, since these files run
         * to hundreds of megabytes.
         */
        fun digestOfFile(path: Path): String {
            val digest = MessageDigest.getInstance("SHA-256")
            path.inputStream().use { stream ->
                val buffer = ByteArray(size = DIGEST_BUFFER_BYTES)
                while (true) {
                    val read = stream.read(buffer)
                    if (read <= 0) {
                        break
                    }
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString(separator = "") { byte -> "%02x".format(byte) }
        }

        /**
         * Opens a vectorizer over a model file and a Hugging Face `tokenizer.json` beside it.
         *
         * The convenience path: it wires ONNX Runtime and the Hugging Face tokenizer, and digests
         * the model file for the fingerprint. Construct the class directly to supply your own
         * [EmbeddingRuntime] or [EmbeddingTokenizer].
         */
        fun open(
            modelPath: Path,
            tokenizerPath: Path,
            pooling: PoolingStrategyEnum = PoolingStrategyEnum.MEAN,
            normalization: NormalizationEnum = NormalizationEnum.L2,
            cache: EmbeddingCache? = null,
            hiddenSize: Int? = null,
        ): OnnxEmbeddingVectorizer {
            val runtime = OnnxEmbeddingRuntime(modelPath = modelPath, hiddenSize = hiddenSize)
            val tokenizer = runCatching {
                HuggingFaceEmbeddingTokenizer(tokenizerPath = tokenizerPath)
            }.getOrElse { cause ->
                runtime.close()
                throw cause
            }
            return OnnxEmbeddingVectorizer(
                runtime = runtime,
                tokenizer = tokenizer,
                modelDigest = digestOfFile(path = modelPath),
                pooling = pooling,
                normalization = normalization,
                cache = cache,
            )
        }
    }
}
