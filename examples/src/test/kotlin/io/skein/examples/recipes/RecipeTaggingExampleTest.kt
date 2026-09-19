package io.skein.examples.recipes

import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class RecipeTaggingExampleTest {

    private val ruleset = RecipeRuleset.load()
    private val example = RecipeTaggingExample(ruleset = ruleset)

    private fun recipe(
        title: String = "bean stew",
        ingredients: String = "beans onion carrot",
        method: String = "simmered",
        minutes: Int = 40,
    ): Recipe {
        return Recipe(
            title = title,
            ingredients = ingredients,
            method = method,
            minutes = minutes,
            origin = "test",
        )
    }

    @Test
    internal fun `the ruleset declares the eight documented labels`() {
        assertEquals(
            expected = listOf(
                "VEGETARIAN", "VEGAN", "GLUTEN_FREE", "ITALIAN", "MEXICAN", "DESSERT", "QUICK", "ONE_POT",
            ),
            actual = ruleset.declaredLabels(),
        )
    }

    @Test
    internal fun `noneOf fires only when no listed term is present`() {
        assertTrue(actual = "VEGETARIAN" in ruleset.label(recipe = recipe(ingredients = "tofu onion")))
        assertTrue(actual = "VEGETARIAN" !in ruleset.label(recipe = recipe(ingredients = "beef onion")))
    }

    @Test
    internal fun `anyOf fires when a term appears in any of its fields`() {
        assertTrue(actual = "ITALIAN" in ruleset.label(recipe = recipe(title = "beef risotto")))
        assertTrue(actual = "ITALIAN" in ruleset.label(recipe = recipe(ingredients = "beef parmesan")))
        assertTrue(actual = "ITALIAN" !in ruleset.label(recipe = recipe()))
    }

    @Test
    internal fun `allOf fires only when every term is present`() {
        assertTrue(actual = "ONE_POT" in ruleset.label(recipe = recipe(method = "one pot")))
        assertTrue(actual = "ONE_POT" !in ruleset.label(recipe = recipe(method = "baked")))
    }

    @Test
    internal fun `atMost fires at and below its limit`() {
        assertTrue(actual = "QUICK" in ruleset.label(recipe = recipe(minutes = 20)))
        assertTrue(actual = "QUICK" !in ruleset.label(recipe = recipe(minutes = 21)))
    }

    /** The hierarchy the example is built to show: everything vegan is also vegetarian. */
    @Test
    internal fun `vegan implies vegetarian on every generated recipe`() {
        RecipeCorpus.training().forEach { recipe ->
            val labels = ruleset.label(recipe = recipe)
            if ("VEGAN" in labels) {
                assertTrue(actual = "VEGETARIAN" in labels, message = "${recipe.title} was vegan but not vegetarian")
            }
        }
    }

    @Test
    internal fun `labels co-occur, which is what makes this a multi-label problem`() {
        val labelled = example.labelled(recipes = RecipeCorpus.training(), grouped = true)

        assertTrue(
            actual = labelled.count { row -> row.labels.size > 1 } > labelled.size / 2,
            message = "most recipes should carry more than one tag",
        )
    }

    /**
     * Groups must have siblings, or grouped cross-validation degenerates into an ungrouped one and
     * the comparison the example prints means nothing.
     */
    @Test
    internal fun `the generated corpus has several recipes per template`() {
        val recipes = RecipeCorpus.training()
        val templates = recipes.groupBy { generated -> generated.origin }

        assertTrue(actual = recipes.size >= 400, message = "only ${recipes.size} recipes")
        assertTrue(actual = templates.size < recipes.size, message = "every recipe was its own template")
        assertTrue(actual = templates.values.all { rows -> rows.size >= 2 })
    }

    @Test
    internal fun `the corpus is identical on every run`() {
        assertEquals(expected = RecipeCorpus.training(), actual = RecipeCorpus.training())
    }

    @Test
    internal fun `the hand-written set uses wording the ruleset does not cover`() {
        val handWritten = RecipeCorpus.handWritten()

        assertTrue(actual = handWritten.size >= 40)
        // Labelled by hand, so the ruleset disagrees with the truth on at least some of them --
        // which is exactly what makes the set worth scoring separately.
        val disagreements = handWritten.count { (recipe, truth) -> ruleset.label(recipe = recipe) != truth }
        assertTrue(actual = disagreements > 0, message = "the rules reproduced every hand label")
    }

    @Test
    internal fun `a trained model tags a recipe it never saw`() {
        val corpus = example.labelled(recipes = RecipeCorpus.training(), grouped = true)
        val model = example.train(corpus = corpus)

        val probe = recipe(title = "tofu risotto", ingredients = "tofu basil rice", method = "baked", minutes = 35)
        val prediction = model.predict(
            features = example.vectorizer().vectorize(text = probe.featureText()),
            threshold = 0.5,
        )

        // Several labels at once, and none of them competing: the point of the capability.
        assertTrue(actual = Label(value = "ITALIAN") in prediction.labels())
        assertTrue(actual = Label(value = "VEGETARIAN") in prediction.labels())
        assertTrue(actual = Label(value = "MEXICAN") !in prediction.labels())
        assertTrue(
            actual = prediction.ranked.sumOf { scored -> scored.probability } > 1.0,
            message = "independent sigmoids should not sum to one",
        )
    }
}
