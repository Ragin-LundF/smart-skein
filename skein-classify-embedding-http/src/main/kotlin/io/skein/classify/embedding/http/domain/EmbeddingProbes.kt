package io.skein.classify.embedding.http.domain

/**
 * Default probe texts for a [io.skein.classify.domain.VectorizerCanary].
 *
 * Short, synthetic, fixed, and deliberately spread across scripts and languages: a model revision
 * that shifts only one language's subspace is invisible to a probe set that is entirely English,
 * and a multilingual encoder is exactly where that happens.
 *
 * **These are stored in the model file in clear text.** That is safe here because they were written
 * for the purpose and say nothing about anybody. Supply your own if you like, but hold to the same
 * rule: never a record from your corpus.
 */
object EmbeddingProbes {

    /** A small multilingual set, enough to catch a changed model without costing much at load. */
    val DEFAULT: List<String> = listOf(
        "the quick brown fox jumps over the lazy dog",
        "ein kurzer satz zur überprüfung des modells",
        "une phrase courte pour vérifier le modèle",
    )
}
