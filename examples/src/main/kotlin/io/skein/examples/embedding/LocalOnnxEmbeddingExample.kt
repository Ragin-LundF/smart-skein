package io.skein.examples.embedding

import io.skein.classify.embedding.onnx.application.EmbeddingCache
import io.skein.classify.embedding.onnx.infrastructure.OnnxEmbeddingVectorizer
import java.nio.file.Path
import kotlin.io.path.exists

/** System properties naming the model files, so the example needs no hard-coded paths. */
private const val MODEL_PROPERTY = "skein.onnx.model"
private const val TOKENIZER_PROPERTY = "skein.onnx.tokenizer"

/**
 * Route A — an embedding model running in this JVM through ONNX Runtime.
 *
 * Point it at an exported model and it trains the recipe tagger on those embeddings and scores it
 * against the hand-written held-out set. Without one it prints how to export a model.
 */
fun runLocalOnnxEmbeddingExample() {
    println("Local ONNX embedding model — route A")
    println("=".repeat(n = 78))

    val modelPath = System.getProperty(MODEL_PROPERTY)?.let { value -> Path.of(value) }
    val tokenizerPath = System.getProperty(TOKENIZER_PROPERTY)?.let { value -> Path.of(value) }

    if (!readable(path = modelPath) || !readable(path = tokenizerPath)) {
        printExportInstructions(modelPath = modelPath, tokenizerPath = tokenizerPath)
        return
    }
    requireNotNull(value = modelPath)
    requireNotNull(value = tokenizerPath)

    OnnxEmbeddingVectorizer.open(
        modelPath = modelPath,
        tokenizerPath = tokenizerPath,
        cache = EmbeddingCache(),
    ).use { vectorizer ->
        println("model     : $modelPath")
        println("tokenizer : $tokenizerPath")
        println("dimension : ${vectorizer.dimension()}")
        println("fingerprint digest: ${vectorizer.fingerprint().configDigest.take(n = 16)}...")

        val metrics = EmbeddingComparison.evaluate(vectorizer = vectorizer)
        EmbeddingComparison.report(embeddingName = modelPath.fileName.toString(), embedding = metrics)

        println()
        println("The fingerprint digests the model file's bytes, so a model replaced in place")
        println("under the same name is refused when a trained model is loaded against it.")
    }
}

private fun readable(path: Path?): Boolean {
    return path != null && path.exists()
}

private fun printExportInstructions(modelPath: Path?, tokenizerPath: Path?) {
    println()
    if (modelPath != null && !modelPath.exists()) {
        println("No model at $modelPath.")
    } else {
        println("No model configured.")
    }
    if (tokenizerPath != null && !tokenizerPath.exists()) {
        println("No tokenizer at $tokenizerPath.")
    }
    println()
    println("Export one with Hugging Face Optimum (Python, one-off):")
    println("     pip install \"optimum[exporters]\"")
    println("     optimum-cli export onnx \\")
    println("       --model intfloat/multilingual-e5-small \\")
    println("       --task feature-extraction \\")
    println("       e5-small-onnx/")
    println()
    println("That writes model.onnx and tokenizer.json. Then:")
    println("     ./gradlew :examples:run --args=\"embedding-onnx\" \\")
    println("       -D$MODEL_PROPERTY=e5-small-onnx/model.onnx \\")
    println("       -D$TOKENIZER_PROPERTY=e5-small-onnx/tokenizer.json")
    println()
    println("E5 models need a 'passage: ' prefix on every input — see the notes in")
    println("     docs/embeddings/onnx-local.md")
}
