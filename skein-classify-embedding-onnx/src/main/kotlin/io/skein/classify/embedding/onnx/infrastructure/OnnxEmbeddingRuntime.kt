package io.skein.classify.embedding.onnx.infrastructure

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import io.skein.classify.embedding.onnx.domain.TokenEmbeddings
import io.skein.classify.embedding.onnx.domain.TokenizedText
import io.skein.classify.embedding.onnx.spi.EmbeddingRuntime
import java.nio.file.Path

/** Input names a sentence encoder may declare. Only those the model actually asks for are bound. */
private const val INPUT_IDS = "input_ids"
private const val ATTENTION_MASK = "attention_mask"
private const val TOKEN_TYPE_IDS = "token_type_ids"

/** A dimension the model leaves symbolic comes back as a negative extent. */
private const val DYNAMIC_DIMENSION = -1L

/**
 * Runs an ONNX sentence-embedding model through ONNX Runtime.
 *
 * **Why ONNX here, when the classifier's own weights are deliberately not ONNX.** The two are
 * different objects. A classifier's weight matrix is sparse, and ONNX's linear operators want dense
 * tensors — measured 574 µs per record against 41 µs, and 97 MB against 17 MB. An embedding model is
 * dense tensor arithmetic from end to end, which is exactly what ONNX Runtime is built for, and it
 * is the only practical way to run a model trained in Python from the JVM.
 *
 * The model's declared inputs drive what is bound: a BERT-family encoder wants `input_ids`,
 * `attention_mask` and `token_type_ids`, while a static embedding model wants only `input_ids`.
 * Binding an input a model did not declare fails, so only the intersection is sent.
 */
class OnnxEmbeddingRuntime(
    modelPath: Path,
    hiddenSize: Int? = null,
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : EmbeddingRuntime {

    private val session: OrtSession = environment.createSession(
        modelPath.toAbsolutePath().toString(),
        OrtSession.SessionOptions(),
    )

    private val outputName: String = session.outputNames.first()

    private val hidden: Int = hiddenSize ?: declaredHiddenSize()

    override fun hiddenSize(): Int {
        return hidden
    }

    override fun encode(batch: List<TokenizedText>): List<TokenEmbeddings> {
        require(value = batch.isNotEmpty()) { "cannot encode an empty batch" }
        val width = batch.first().ids.size
        require(value = batch.all { row -> row.ids.size == width }) {
            "every row of a batch must be padded to the same length; " +
                "got ${batch.map { row -> row.ids.size }.distinct()}"
        }
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            if (INPUT_IDS in session.inputNames) {
                inputs[INPUT_IDS] = OnnxTensor.createTensor(
                    environment,
                    batch.map { row -> row.ids }.toTypedArray(),
                )
            }
            if (ATTENTION_MASK in session.inputNames) {
                inputs[ATTENTION_MASK] = OnnxTensor.createTensor(
                    environment,
                    batch.map { row -> row.attentionMask }.toTypedArray(),
                )
            }
            if (TOKEN_TYPE_IDS in session.inputNames) {
                // A single-sequence encoder wants all zeros here; the segment distinction only
                // matters for sentence-pair models, which this adapter does not address.
                inputs[TOKEN_TYPE_IDS] = OnnxTensor.createTensor(
                    environment,
                    Array(size = batch.size) { LongArray(size = width) },
                )
            }
            return session.run(inputs).use { result ->
                @Suppress("UNCHECKED_CAST")
                val output = result.get(outputName).orElseThrow().value as Array<Array<FloatArray>>
                output.mapIndexed { index, sequence ->
                    TokenEmbeddings(vectors = sequence, attentionMask = batch[index].attentionMask)
                }
            }
        } finally {
            inputs.values.forEach { tensor -> tensor.close() }
        }
    }

    override fun close() {
        session.close()
    }

    /**
     * The hidden size the model declares on its output, or a failure naming the fix.
     *
     * A model may leave the last dimension symbolic, in which case it genuinely cannot be known
     * without running the model. Saying so and asking for it is better than probing with a
     * fabricated input, which would succeed on some architectures and produce a confusing failure
     * on others.
     */
    private fun declaredHiddenSize(): Int {
        val info = session.outputInfo[outputName]?.info as? TensorInfo
        val declared = info?.shape?.lastOrNull() ?: DYNAMIC_DIMENSION
        check(value = declared > 0L) {
            "model output '$outputName' leaves its hidden size dynamic; " +
                "pass hiddenSize explicitly (it is the model's embedding dimension, e.g. 384 or 768)"
        }
        return declared.toInt()
    }
}
