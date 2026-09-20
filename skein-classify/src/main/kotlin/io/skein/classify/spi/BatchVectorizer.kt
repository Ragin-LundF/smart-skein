package io.skein.classify.spi

import io.skein.classify.domain.FeatureVector

/**
 * A [Vectorizer] for which featurising many texts at once is **materially cheaper** than
 * featurising them one at a time.
 *
 * Not every vectorizer needs this. [io.skein.classify.application.HashingVectorizer] hashes each
 * record in microseconds with no per-call overhead worth amortising, so a loop is already optimal
 * and it deliberately does not implement this port. The port exists for the vectorizers where the
 * difference is not a percentage:
 *
 * | Vectorizer | One at a time | Batched |
 * |---|---|---|
 * | An ONNX encoder | one inference call per record | one call per batch, which is what a GPU exists for |
 * | An embedding **service** | one HTTP round trip per record | one request per batch |
 *
 * At a few thousand records that is the difference between a run that finishes and one that does
 * not. It is the reason this is a capability a caller can *detect* rather than a method with a
 * default: code that is about to featurise a whole corpus can reasonably want to know whether it
 * is about to make one network call or four hundred thousand.
 *
 * **Implementations must return one vector per input, in the order the inputs were given.** A
 * silently permuted batch attaches every vector to the wrong record, which does not throw and does
 * not look wrong in the output — the same class of failure that
 * [io.skein.classify.domain.VectorizerFingerprint] exists to prevent.
 */
interface BatchVectorizer : Vectorizer {

    /**
     * Featurises [texts] in one pass, returning one [FeatureVector] per input **in input order**.
     *
     * An empty list in, an empty list out, without contacting anything.
     */
    fun vectorizeAll(texts: List<String>): List<FeatureVector>
}

/**
 * Featurises [texts] using batching when this vectorizer supports it, and a plain loop when it
 * does not.
 *
 * This is what library code and callers holding a `Vectorizer` should use for a whole corpus.
 * A [BatchVectorizer]'s own `vectorizeAll` member takes precedence over this extension whenever
 * the static type says so, and the two behave identically — the extension only exists so that no
 * caller has to write the `as? BatchVectorizer` cast.
 *
 * Note that it returns every vector at once. For an embedding vectorizer over a large corpus that
 * is a real amount of memory; chunk the call site if the corpus does not fit.
 */
fun Vectorizer.vectorizeAll(texts: List<String>): List<FeatureVector> {
    if (this is BatchVectorizer) {
        return vectorizeAll(texts = texts)
    }
    return texts.map { text -> vectorize(text = text) }
}
