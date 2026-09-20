package io.skein.examples.embedding

import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.infrastructure.HttpEmbeddingVectorizer

/**
 * Route B — embeddings from an external service, here LM Studio.
 *
 * Run it with a local server up and it trains the recipe tagger on service-provided embeddings and
 * scores it against the hand-written held-out set. Run it without one and it prints the setup.
 */
fun runExternalEmbeddingExample() {
    val config = EmbeddingServiceConfig.lmStudioMultilingualE5Small()
    val vectorizer = HttpEmbeddingVectorizer(config = config)

    println("External embedding service — route B")
    println("=".repeat(n = 78))
    println("endpoint : ${config.embeddingsUrl()}")
    println("model    : ${config.model}  (revision ${config.modelRevision})")
    println("prefix   : '${config.inputPrefix}'")

    if (!vectorizer.isReachable()) {
        printSetupInstructions(config = config)
        return
    }

    println("dimension: ${vectorizer.dimension()}")
    println("fingerprint digest: ${vectorizer.fingerprint().configDigest.take(n = 16)}...")

    val metrics = EmbeddingComparison.evaluate(vectorizer = vectorizer)
    EmbeddingComparison.report(embeddingName = config.model, embedding = metrics)

    println()
    println("What the fingerprint cannot do, and what the canary does instead")
    println("-".repeat(n = 78))
    println("   The fingerprint pins the model name, your declared revision, the prefix and the")
    println("   width. It cannot detect the served weights changing, because nothing in the")
    println("   protocol reveals them. Bump EmbeddingServiceConfig.modelRevision whenever you")
    println("   update the model — but that is a declaration, and nothing enforces it.")
    println()
    reportCanary(vectorizer = vectorizer)
}

/**
 * The part that actually checks. The probes are embedded once here and again inside
 * `ModelStore.saveMultiLabel`, and re-embedded on every `loadMultiLabel` — so a model changed on
 * the server stops the load instead of quietly returning wrong labels.
 */
private fun reportCanary(vectorizer: HttpEmbeddingVectorizer) {
    val canary = vectorizer.canary()
    if (canary == null) {
        println("   No canary configured. Set EmbeddingServiceConfig.canaryProbes to enable it.")
        return
    }
    val reEmbedded = canary.probes.map { probe -> vectorizer.vectorize(text = probe).values }
    val worst = canary.worstDrift(observed = reEmbedded)
    println("   Canary: ${canary.probes.size} probes, tolerance ${canary.tolerance}.")
    println("   Worst drift right now: ${worst.distance} — a changed model would push this past")
    println("   the tolerance and loadMultiLabel would throw VectorizerCanaryException.")
}

private fun printSetupInstructions(config: EmbeddingServiceConfig) {
    println()
    println("No embedding service answered at ${config.embeddingsUrl()}.")
    println()
    println("To run this example with LM Studio:")
    println("  1. Install LM Studio and open the Discover tab.")
    println("  2. Download an embedding model. 'multilingual-e5-small' is a good default:")
    println("     384 dimensions, ~100 languages, about half a gigabyte, comfortable on a CPU.")
    println("  3. Open the Developer tab, load the model, and start the server (default port 1234).")
    println("  4. Confirm the model id LM Studio reports and set it in EmbeddingServiceConfig.model.")
    println("  5. Re-run:  ./gradlew :examples:run --args=\"embedding-service\"")
    println()
    println("Ollama works the same way:")
    println("     ollama pull snowflake-arctic-embed2  &&  ollama serve")
    println("     baseUrl = \"http://localhost:11434/v1\"")
    println()
    println("Full setup, model choices and the training workflow:")
    println("     docs/embeddings/external-service.md")
}
