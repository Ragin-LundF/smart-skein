package io.skein.classify.domain

/**
 * The stable identity of a vectorizer **and its configuration**, persisted with a model and checked
 * when one is loaded.
 *
 * **What this prevents.** A model trained with one featurisation and scored with another does not
 * fail. It produces confident nonsense: the indices still resolve, the weights still multiply, and
 * the labels come back wrong with no exception and no log line. That failure is invisible in
 * production and nearly impossible to diagnose from the output. Two vectorizers sharing a
 * fingerprint must therefore produce identical vectors for identical input, and
 * [io.skein.classify.application.ModelStore] treats a mismatch as fatal rather than as a warning.
 *
 * [configDigest] must cover *everything* that changes the output vector — for keyed hashing the key,
 * the width and the n-gram ranges; for a fitted vocabulary a digest of the terms and their weights;
 * for an external embedding model **the digest of the model file itself**, not its name or version
 * string. A model file swapped in place under an unchanged name is precisely the failure this
 * exists to catch.
 */
data class VectorizerFingerprint(
    val kind: String,
    val dimension: Int,
    val configDigest: String,
) {

    init {
        require(value = kind.isNotBlank()) { "kind must not be blank" }
        require(value = dimension > 0) { "dimension must be positive, got $dimension" }
        require(value = configDigest.isNotBlank()) { "configDigest must not be blank" }
    }
}
