package io.skein.examples.recipes

import kotlinx.serialization.Serializable

/**
 * One rule: a label, and one condition block per operator.
 *
 * Four operators against named fields is the entire grammar — deliberately flatter than a real
 * rule engine, with no preconditions, no priority ordering and no per-tenant overrides. The
 * simplicity is the lesson: **the library never sees this file.** Rules exist only to produce
 * training data, so any rule format, any language and any legacy engine works exactly the same way.
 *
 * A rule fires when every block it declares is satisfied. An empty block is no constraint at all.
 */
@Serializable
data class RecipeRule(
    val label: String,
    /** Fires when **any** listed term appears in its field. */
    val anyOf: Map<String, List<String>> = emptyMap(),
    /** Fires when **every** listed term appears in its field. */
    val allOf: Map<String, List<String>> = emptyMap(),
    /** Fires when **no** listed term appears in its field. */
    val noneOf: Map<String, List<String>> = emptyMap(),
    /** Fires when the named numeric field is at or below the limit. */
    val atMost: Map<String, Double> = emptyMap(),
) {

    /** Whether this rule fires for [recipe]. */
    fun matches(recipe: Recipe): Boolean {
        return matchesAnyOf(recipe = recipe) &&
            matchesAllOf(recipe = recipe) &&
            matchesNoneOf(recipe = recipe) &&
            matchesAtMost(recipe = recipe)
    }

    private fun matchesAnyOf(recipe: Recipe): Boolean {
        if (anyOf.isEmpty()) {
            return true
        }
        return anyOf.any { (field, terms) ->
            terms.any { term -> contains(recipe = recipe, field = field, term = term) }
        }
    }

    private fun matchesAllOf(recipe: Recipe): Boolean {
        return allOf.all { (field, terms) ->
            terms.all { term -> contains(recipe = recipe, field = field, term = term) }
        }
    }

    private fun matchesNoneOf(recipe: Recipe): Boolean {
        return noneOf.none { (field, terms) ->
            terms.any { term -> contains(recipe = recipe, field = field, term = term) }
        }
    }

    private fun matchesAtMost(recipe: Recipe): Boolean {
        return atMost.all { (field, limit) -> (recipe.number(name = field) ?: Double.MAX_VALUE) <= limit }
    }

    private fun contains(recipe: Recipe, field: String, term: String): Boolean {
        return recipe.text(name = field)?.contains(other = term, ignoreCase = true) == true
    }
}
