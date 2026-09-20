package io.skein.examples.localai

import io.skein.classify.application.ModelStore
import io.skein.classify.application.VectorizerCanaryException
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.Schema
import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.embedding.http.domain.EmbeddingProbes
import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.infrastructure.HttpEmbeddingVectorizer
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier
import io.skein.examples.recipes.RecipeCorpus
import io.skein.examples.recipes.RecipeRuleset
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteIfExists

/** Where a local OpenAI-compatible server usually lives. Override with `-Dskein.localai.url=...`. */
private const val DEFAULT_BASE_URL = "http://localhost:1234/v1"

private val RULE = "-".repeat(n = 78)

private const val BATCH_SIZE = 32
private const val TIMEOUT_SECONDS = 180L
private const val SWAPPED_SEED = 0xDECAFL

/**
 * **Route B in production: running a classifier against a local AI server safely.**
 *
 * The [embedding-service][io.skein.examples.embedding.runExternalEmbeddingExample] example answers
 * *"are embeddings worth it for my data?"*. This one answers the question that comes next:
 * *"how do I run this without it silently breaking?"*
 *
 * An embedding service is the one vectorizer whose identity cannot be verified from the outside.
 * It returns vectors; nothing in the protocol says which weights produced them. A model updated on
 * the server produces different vectors for the same text, a classifier trained against the old
 * ones keeps scoring without error, and the labels are quietly wrong. This walks the four things
 * that stop that:
 *
 * 1. **Discover the model** rather than hard-coding an id that does not exist ([ModelDiscovery]).
 * 2. **Calibrate the tolerance** against your own model's noise ([CanaryCalibrator]).
 * 3. **Train, save and reload** with a canary in the file.
 * 4. **Watch it catch a real swap** — staged live, because a [StubEmbeddingServer] can change its
 *    weights mid-run and a real one cannot be asked to.
 *
 * Runs against a real server when one answers, and against an in-process stub when none does, so
 * it always has something to show.
 *
 * ```bash
 * ./gradlew :examples:run --args="localai"
 * ./gradlew :examples:run --args="localai" -Dskein.localai.url=http://localhost:11434/v1
 * ./gradlew :examples:run --args="localai" -Dskein.localai.model=nomic-embed-text-v1.5
 * ```
 */
fun runLocalAiExample() {
    println("Local AI — operating a classifier against an embedding service")
    println("=".repeat(n = 78))

    val baseUrl = System.getProperty("skein.localai.url") ?: DEFAULT_BASE_URL
    val preferred = System.getProperty("skein.localai.model")
    val discovered = ModelDiscovery.firstEmbeddingModel(baseUrl = baseUrl, preferred = preferred)

    if (discovered != null) {
        println("Found a real service at $baseUrl")
        println(RULE)
        walkThrough(baseUrl = baseUrl, model = discovered, live = true)
        return
    }

    println(ModelDiscovery.diagnose(baseUrl = baseUrl))
    println()
    println("Continuing against an in-process stub so there is still something to see.")
    println("Everything below is real code on a real socket — only the weights are fake.")
    println(RULE)

    StubEmbeddingServer().start().use { stub ->
        walkThrough(
            baseUrl = stub.baseUrl,
            model = DiscoveredModel(id = "stub-embedding-model", dimension = stub.dimension),
            live = false,
            stub = stub,
        )
    }
}

private fun walkThrough(
    baseUrl: String,
    model: DiscoveredModel,
    live: Boolean,
    stub: StubEmbeddingServer? = null,
) {
    val config = pinModel(baseUrl = baseUrl, model = model)
    calibrateTolerance(config = config)

    val file = Files.createTempFile("localai-example", ".skein")
    try {
        trainSaveAndReload(config = config, model = model, file = file)
        stageAModelSwap(config = config, file = file, stub = stub)
    } finally {
        file.deleteIfExists()
    }

    printProductionNotes(live = live)
}

/** Step 1 — turn a discovered model into the configuration you would actually deploy. */
private fun pinModel(baseUrl: String, model: DiscoveredModel): EmbeddingServiceConfig {
    step(number = 1, title = "Discover the model, then pin it")
    println("   id        : ${model.id}")
    println("   dimension : ${model.dimension}")
    println()
    println("   The id is not guessable — it is whatever the server called the download. The same")
    println("   multilingual-e5-small weights are 'multilingual-e5-small-mlx' on a machine that")
    println("   pulled the MLX build. Discover it once at startup, log it, then pin it in config.")

    return EmbeddingServiceConfig(
        baseUrl = baseUrl,
        model = model.id,
        modelRevision = "${model.id}@1",
        dimension = model.dimension,
        canaryProbes = EmbeddingProbes.DEFAULT,
        batchSize = BATCH_SIZE,
        timeoutSeconds = TIMEOUT_SECONDS,
    )
}

/** Step 2 — find out how far this model's vectors move when nothing has changed. */
private fun calibrateTolerance(config: EmbeddingServiceConfig) {
    step(number = 2, title = "Calibrate the canary tolerance against this model")
    val calibrator = CanaryCalibrator()
    val floor = calibrator.noiseFloor(
        vectorizer = HttpEmbeddingVectorizer(config = config),
        probes = EmbeddingProbes.DEFAULT,
        reader = HttpEmbeddingVectorizer(config = config),
    )
    println("   noise floor : ${calibrator.verdict(reading = floor)}")
    println()
    println("   That is how far the vectors move when nothing has changed, measured through a")
    println("   second, independent client. It has to sit far below the tolerance, or every load")
    println("   fails on arithmetic. Probes are embedded one per request, never batched, because")
    println("   batch composition perturbs accumulation order all by itself.")
}

/** Step 3 — the ordinary lifecycle, with the canary riding along in the file. */
private fun trainSaveAndReload(config: EmbeddingServiceConfig, model: DiscoveredModel, file: Path) {
    step(number = 3, title = "Train, save, reload — the canary travels in the file")
    val vectorizer = HttpEmbeddingVectorizer(config = config)
    val recipes = RecipeCorpus.training()
    val ruleset = RecipeRuleset.load()
    val texts = recipes.map { recipe -> recipe.featureText() }

    val vectors = vectorizer.vectorizeAll(texts = texts)
    val batches = (texts.size + config.batchSize - 1) / config.batchSize
    println("   embedded ${texts.size} recipes in $batches requests, not ${texts.size} —")
    println("   HttpEmbeddingVectorizer is a BatchVectorizer, so vectorizeAll batches for free")

    val trained = LbfgsMultiLabelLearner(featureCount = model.dimension, keepFraction = 1.0).fit(
        observations = recipes.indices.map { index ->
            MultiLabeledFeatures(
                features = vectors[index],
                labels = ruleset.label(recipe = recipes[index]).map { value -> Label(value = value) }.toSet(),
            )
        },
    ) as MultiLabelLogisticClassifier

    val schema = Schema.define {
        text(name = "recipe")
        label(name = "tags")
    }
    ModelStore.saveMultiLabel(path = file, schema = schema, model = trained, vectorizer = vectorizer)
    println("   saved ${Files.size(file)} bytes, canary included")

    val loaded = ModelStore.loadMultiLabel(path = file, vectorizer = HttpEmbeddingVectorizer(config = config))
    println("   reloaded, canary verified over ${loaded.canary?.probes?.size} probes")
    println()
    println("   Loading now performs network I/O — one round trip per probe — and can fail")
    println("   because the service is down. That is the trade: a loud failure at load instead")
    println("   of a silent one at inference. loadMultiLabel(..., verifyCanary = false) opts out")
    println("   when you would rather serve than be certain.")
}

/**
 * Step 4 — the failure the canary exists for, staged live when running against the stub.
 *
 * A real server cannot be asked to change its weights mid-run, which is exactly why
 * [StubEmbeddingServer] is worth keeping around.
 */
private fun stageAModelSwap(config: EmbeddingServiceConfig, file: Path, stub: StubEmbeddingServer?) {
    step(number = 4, title = "What happens when the served model changes")
    if (stub == null) {
        println("   Not staged against a live service — swapping a real model means loading a")
        println("   different one. To see it for real:")
        println()
        println("     lms unload --all && lms load <a-different-embedding-model> -y")
        println("     ./gradlew :examples:run --args=\"localai\"")
        println()
        println("   The reload above would then throw VectorizerCanaryException. Re-run this")
        println("   example with no server reachable to watch it happen against the stub.")
        return
    }

    val before = HttpEmbeddingVectorizer(config = config).fingerprint()
    stub.seed = SWAPPED_SEED
    val after = HttpEmbeddingVectorizer(config = config)

    println("   The stub is now serving different weights. Same URL, same model name, same")
    println("   declared revision, same width.")
    println("   fingerprint still matches : ${after.fingerprint() == before}   <- why a fingerprint is not enough")

    runCatching { ModelStore.loadMultiLabel(path = file, vectorizer = after) }
        .onFailure { cause ->
            if (cause is VectorizerCanaryException) {
                val drift = "%.6f".format(cause.drift)
                println("   load refused    : drift $drift > ${VectorizerCanary.DEFAULT_TOLERANCE}")
                println("   worst probe     : \"${cause.probe}\"")
            }
        }
        .onSuccess { println("   load SUCCEEDED — the canary did not fire, which would be a bug") }
}

/** Step 5 — the operational checklist this example exists to hand over. */
private fun printProductionNotes(live: Boolean) {
    step(number = 5, title = "Taking this to production")
    println("   1. Pin the model id and bump modelRevision whenever you change the served model.")
    println("   2. Keep canaryProbes set. They are short synthetic sentences you choose — never")
    println("      records from your corpus, because they are stored in the model file in clear.")
    println("   3. Measure your own noise floor (step 2) before trusting the default tolerance.")
    println("   4. Re-check drift periodically in a long-running service: hold")
    println("      LoadedMultiLabelModel.canary and call CanaryCalibrator.driftAgainst on a timer.")
    println("      A service can be updated while you are holding the model open.")
    println("   5. Prefer route A (skein-classify-embedding-onnx) where the choice is open: it")
    println("      hashes the model file's bytes, which makes a swap impossible rather than")
    println("      merely detectable.")
    if (!live) {
        println()
        println("   Re-run with a real server for figures that mean something:")
        println("     lms load <an-embedding-model> -y")
        println("     ./gradlew :examples:run --args=\"localai\"")
    }
}

private fun step(number: Int, title: String) {
    println()
    println("$number. $title")
    println(RULE)
}
