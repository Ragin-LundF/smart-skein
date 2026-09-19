package io.skein.examples.recipes

/**
 * One recipe: three text fields and a numeric one, which is enough to exercise a mixed
 * [io.skein.classify.domain.Schema] properly.
 *
 * [origin] names where this row came from — the generator template that produced it, or `"hand"`
 * for the hand-written held-out set. It is the grouping key for cross-validation, and it is
 * metadata about the *data*, never a feature: nothing derived from it reaches the model.
 */
data class Recipe(
    val title: String,
    val ingredients: String,
    val method: String,
    val minutes: Int,
    val origin: String,
) {

    /** The text of [name], for the rule interpreter. Comparison is case-insensitive throughout. */
    fun text(name: String): String? {
        return when (name) {
            "title" -> title
            "ingredients" -> ingredients
            "method" -> method
            else -> null
        }
    }

    /** The numeric value of [name], for the rule interpreter. */
    fun number(name: String): Double? {
        return if (name == "minutes") minutes.toDouble() else null
    }

    /**
     * What the classifier sees. The numeric field is rendered into the text because feature hashing
     * works on text — a real pipeline would use
     * [io.skein.classify.application.RecordMapper] over a `Schema`, which does the same thing.
     */
    fun featureText(): String {
        return "$title $ingredients $method $minutes minutes"
    }
}
