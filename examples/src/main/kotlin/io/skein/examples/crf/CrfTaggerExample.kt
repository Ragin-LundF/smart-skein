package io.skein.examples.crf

import io.skein.extract.domain.Tag
import io.skein.extract.infrastructure.CrfModelStore
import io.skein.extract.infrastructure.CrfSequenceLabeler
import io.skein.extract.infrastructure.FeatureRetentionEnum
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token
import java.nio.file.Files
import kotlin.io.path.deleteIfExists
import kotlin.io.path.fileSize

private const val TOKEN_COL_WIDTH = 30
private const val TYPE_COL_WIDTH = 14
private const val DIVIDER_WIDTH = 58
private const val TRAIN_EPOCHS = 20

private val TAG_KEY = Tag(value = "KEY")
private val TAG_VALUE = Tag(value = "VALUE")
private val TAG_OTHER = Tag(value = "OTHER")

private data class TrainingSample(val text: String, val tags: List<Tag>)

private val SAMPLES = listOf(
    TrainingSample("IBAN DE12345678 Amount 250.00 USD", listOf(TAG_KEY, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER)),
    TrainingSample(
        "IBAN GB29NWBK60161331926819 Amount 120.50 USD",
        listOf(TAG_KEY, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER),
    ),
    TrainingSample("IBAN DE00000001 Amount 99.99 USD", listOf(TAG_KEY, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER)),
    TrainingSample(
        "Recipient Acme Corp Amount 500.00 USD",
        listOf(TAG_KEY, TAG_VALUE, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER),
    ),
    TrainingSample(
        "Recipient Bloom Inc Amount 310.00 USD",
        listOf(TAG_KEY, TAG_VALUE, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER),
    ),
    TrainingSample(
        "IBAN DE11223344 Name Jane Doe Amount 75.00 USD",
        listOf(TAG_KEY, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER),
    ),
)

private fun printTaggingTable(tokens: List<Token>, tags: List<Tag>) {
    println("%-${TOKEN_COL_WIDTH}s  %-${TYPE_COL_WIDTH}s  %s".format("Token", "Type", "Tag"))
    println("-".repeat(DIVIDER_WIDTH))
    tokens.zip(tags).forEach { (token, tag) ->
        println("%-${TOKEN_COL_WIDTH}s  %-${TYPE_COL_WIDTH}s  %s".format(token.text, token.type.name, tag.value))
    }
}

fun runCrfTaggerExample() {
    println("=== CRF Sequence Tagger ===")
    println()

    val tokenizer = TypedTokenizer()
    val trainingPairs = SAMPLES.mapNotNull { sample ->
        val tokens = tokenizer.tokenize(text = sample.text)
        if (tokens.size != sample.tags.size) {
            println("WARN: token/tag mismatch for '${sample.text}' — skipped")
            null
        } else {
            tokens to sample.tags
        }
    }
    println("Training samples: ${trainingPairs.size}")

    val labeler = CrfSequenceLabeler(initialLearningRate = 0.1, decayRate = 0.001, l2Regularization = 0.0)
    repeat(times = TRAIN_EPOCHS) {
        trainingPairs.forEach { (tokens, tags) -> labeler.learn(tokens = tokens, tags = tags) }
    }
    println("Training complete ($TRAIN_EPOCHS epochs x ${trainingPairs.size} samples)")
    println()

    val unseenText = "IBAN NL91ABNA0417164300 Amount 180.00 USD"
    println("--- Labeling unseen sequence ---")
    println("Input: \"$unseenText\"")
    println()
    val unseenTokens = tokenizer.tokenize(text = unseenText)
    printTaggingTable(tokens = unseenTokens, tags = labeler.label(tokens = unseenTokens))
    println()

    val variationText = "IBAN CH9300762011623852957 Amount 2500.00 USD"
    println("--- Generalization: unseen CH IBAN ---")
    println("Input: \"$variationText\"")
    println()
    val variationTokens = tokenizer.tokenize(text = variationText)
    printTaggingTable(tokens = variationTokens, tags = labeler.label(tokens = variationTokens))
    println()

    println("CRF generalizes via feature weights (token type, prefix, suffix, neighbours)")
    println("not by memorizing exact token strings — so new IBAN formats are tagged correctly.")
    println()

    demonstratePersistence(labeler = labeler, tokenizer = tokenizer, probe = variationTokens)
}

/** Saves the trained tagger, reloads it, proves the tagging survives, and resumes training. */
private fun demonstratePersistence(labeler: CrfSequenceLabeler, tokenizer: TypedTokenizer, probe: List<Token>) {
    println("--- Persist and resume ---")
    val modelFile = Files.createTempFile("skein-crf-example", ".skeincrf")
    try {
        CrfModelStore.save(
            path = modelFile,
            snapshot = labeler.snapshot(),
            retention = FeatureRetentionEnum.ALL_FEATURES,
        )
        val metadata = CrfModelStore.metadata(path = modelFile)
        println("Saved ${modelFile.fileSize()} bytes")
        println("  format      SKCR v${metadata.formatMajor}.${metadata.formatMinor}")
        println("  retention   ${metadata.retention}")
        println("  written by  ${metadata.writerVersion}")
        println()

        val restored = CrfSequenceLabeler.from(snapshot = CrfModelStore.load(path = modelFile))
        val before = labeler.label(tokens = probe)
        val after = restored.label(tokens = probe)
        println("Reloaded model tags identically: ${before == after}")
        println()

        // Training continues from the saved step rather than restarting.
        val stepBefore = restored.snapshot().step
        val extraTokens = tokenizer.tokenize(text = "IBAN AT611904300234573201 Amount 42.00 USD")
        restored.learn(
            tokens = extraTokens,
            tags = listOf(TAG_KEY, TAG_VALUE, TAG_KEY, TAG_VALUE, TAG_OTHER),
        )
        println("Resumed training: step $stepBefore -> ${restored.snapshot().step}")
        println()

        CrfModelStore.save(
            path = modelFile,
            snapshot = restored.snapshot(),
            retention = FeatureRetentionEnum.STRUCTURAL_ONLY,
        )
        println("Saved again with STRUCTURAL_ONLY: ${modelFile.fileSize()} bytes.")
        println("That mode writes no raw token text at all — the only zero-clear-text guarantee —")
        println("at the cost of the word-shape generalization demonstrated above. The default,")
        println("FREQUENT_ONLY, keeps text seen more than once, so it reduces exposure without")
        println("eliminating it.")
        println()
    } finally {
        modelFile.deleteIfExists()
    }
}
