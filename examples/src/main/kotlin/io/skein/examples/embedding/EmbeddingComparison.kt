package io.skein.examples.embedding

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.application.MultiLabelEvaluator
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabelMetrics
import io.skein.classify.domain.MultiLabelMetricsFactory
import io.skein.classify.domain.MultiLabelOutcome
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner
import io.skein.classify.spi.Vectorizer
import io.skein.examples.recipes.Recipe
import io.skein.examples.recipes.RecipeCorpus
import io.skein.examples.recipes.RecipeRuleset
import java.util.Locale

private const val THRESHOLD = 0.5
private const val HASHING_FEATURES = 1 shl 15

/**
 * Trains the recipe tagger twice — once on hashed n-grams, once on embeddings — and scores both
 * against the hand-written held-out set.
 *
 * That set is the whole point of the comparison. It uses wording the ruleset never mentions
 * ("aubergine", "courgette", "cacio e pepe"), so it measures the one thing an embedding is supposed
 * to buy: recognising that a word means something close to a word the model was trained on. Hashed
 * n-grams match text literally and cannot do this; a sentence encoder can, and this prints by how
 * much on your model.
 */
object EmbeddingComparison {

    /** Scores [vectorizer] on the hand-written held-out set after training on the generated corpus. */
    fun evaluate(
        vectorizer: Vectorizer,
        batched: ((List<String>) -> List<FeatureVector>)? = null,
    ): MultiLabelMetrics {
        val ruleset = RecipeRuleset.load()
        val training = RecipeCorpus.training()
        val trainingTexts = training.map { recipe -> recipe.featureText() }
        val trainingVectors = batched?.invoke(trainingTexts)
            ?: trainingTexts.map { text -> vectorizer.vectorize(text = text) }

        val model = LbfgsMultiLabelLearner(
            featureCount = vectorizer.dimension(),
            keepFraction = 1.0,
        ).fit(
            observations = training.indices.map { index ->
                MultiLabeledFeatures(
                    features = trainingVectors[index],
                    labels = ruleset.label(recipe = training[index])
                        .map { value -> Label(value = value) }
                        .toSet(),
                )
            },
        )

        val handWritten = RecipeCorpus.handWritten()
        val handTexts = handWritten.map { (recipe, _) -> recipe.featureText() }
        val handVectors = batched?.invoke(handTexts) ?: handTexts.map { text -> vectorizer.vectorize(text = text) }

        return MultiLabelMetricsFactory.from(
            outcomes = handWritten.indices.map { index ->
                MultiLabelOutcome(
                    expected = handWritten[index].second.map { value -> Label(value = value) }.toSet(),
                    prediction = model.predict(features = handVectors[index], threshold = THRESHOLD),
                )
            },
        )
    }

    /** The hashed-n-gram baseline the embedding is measured against. */
    fun hashingBaseline(): MultiLabelMetrics {
        return evaluate(
            vectorizer = HashingVectorizer(
                config = HashingConfig(key0 = 0x5EEDL, key1 = 0xCAFEL, numFeatures = HASHING_FEATURES),
            ),
        )
    }

    /** Prints both scores with the difference spelled out. */
    fun report(embeddingName: String, embedding: MultiLabelMetrics) {
        val hashing = hashingBaseline()
        println()
        println("Hand-written held-out set — wording the rules never mention")
        println("-".repeat(n = 78))
        println("   ${"hashed n-grams".padEnd(length = 34)} ${row(metrics = hashing)}")
        println("   ${embeddingName.padEnd(length = 34)} ${row(metrics = embedding)}")
        println()
        val delta = embedding.micro.f1 - hashing.micro.f1
        val favours = if (delta >= 0) "the embedding" else "hashing"
        println("   micro-F1 difference: ${format(value = delta)} in favour of $favours")
        println()
        println("   An embedding earns its cost here or nowhere. On wording the training data")
        println("   already contains, hashed n-grams are as good and about 25-75x faster.")
    }

    private fun row(metrics: MultiLabelMetrics): String {
        return "P ${format(value = metrics.micro.precision)}  R ${format(value = metrics.micro.recall)}  " +
            "micro-F1 ${format(value = metrics.micro.f1)}  macro-F1 ${format(value = metrics.macro.f1)}"
    }

    private fun format(value: Double): String {
        return String.format(Locale.ROOT, "%+.3f", value).removePrefix(prefix = "+")
    }

    /** A few hand-written recipes, for printing individual predictions. */
    fun sampleRecipes(): List<Recipe> {
        return RecipeCorpus.handWritten().take(n = 5).map { (recipe, _) -> recipe }
    }

    /** Recall at a few depths, which is the shape human review actually uses. */
    fun rankedRecall(vectorizer: Vectorizer): Map<Int, Double> {
        val evaluator = MultiLabelEvaluator()
        val ruleset = RecipeRuleset.load()
        val handWritten = RecipeCorpus.handWritten()
        val training = RecipeCorpus.training()
        val model = LbfgsMultiLabelLearner(featureCount = vectorizer.dimension(), keepFraction = 1.0).fit(
            observations = training.map { recipe ->
                MultiLabeledFeatures(
                    features = vectorizer.vectorize(text = recipe.featureText()),
                    labels = ruleset.label(recipe = recipe).map { value -> Label(value = value) }.toSet(),
                )
            },
        )
        return evaluator.recallAtK(
            outcomes = handWritten.map { (recipe, truth) ->
                MultiLabelOutcome(
                    expected = truth.map { value -> Label(value = value) }.toSet(),
                    prediction = model.predict(
                        features = vectorizer.vectorize(text = recipe.featureText()),
                        threshold = THRESHOLD,
                    ),
                )
            },
        )
    }
}
