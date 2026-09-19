package io.skein.examples.recipes

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * The ruleset, and the whole of the "rule engine" this example replaces.
 *
 * A dozen lines of interpretation, living in the example rather than in the library, because the
 * library genuinely does not care: rules produce labels, labels produce training data, and the
 * model learns from the labels. That boundary is the point of the example.
 */
@Serializable
data class RecipeRuleset(val version: Int, val rules: List<RecipeRule>) {

    /** Every label whose rule fires for [recipe]. May be empty, and that is meaningful. */
    fun label(recipe: Recipe): Set<String> {
        return rules.filter { rule -> rule.matches(recipe = recipe) }
            .map { rule -> rule.label }
            .toSet()
    }

    /** Labels the ruleset can produce at all, in declaration order. */
    fun declaredLabels(): List<String> {
        return rules.map { rule -> rule.label }.distinct()
    }

    companion object {

        private val JSON = Json { ignoreUnknownKeys = true }

        /** Loads the ruleset shipped beside this example. */
        fun load(): RecipeRuleset {
            val stream = RecipeRuleset::class.java.getResourceAsStream("/recipe-rules.json")
            checkNotNull(value = stream) { "recipe-rules.json is missing from the example resources" }
            return stream.use { input ->
                JSON.decodeFromString(deserializer = serializer(), string = input.readBytes().decodeToString())
            }
        }
    }
}
