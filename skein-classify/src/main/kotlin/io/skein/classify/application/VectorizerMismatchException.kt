package io.skein.classify.application

import io.skein.classify.domain.VectorizerFingerprint

/**
 * Thrown when a model is loaded with a vectorizer other than the one it was trained with.
 *
 * **Why this is fatal rather than a warning.** Featurisation mismatch does not fail loudly on its
 * own: the feature indices still resolve, the weights still multiply, and the model returns
 * confident labels that are simply wrong. There is no error to notice, nothing in the output looks
 * unusual, and the damage is silent for as long as the deployment lasts. A log line would be read
 * after the fact, if at all.
 */
class VectorizerMismatchException(
    val expected: VectorizerFingerprint,
    val actual: VectorizerFingerprint,
) : IllegalStateException(
    "model was trained with vectorizer ${expected.kind}/${expected.dimension} " +
        "(digest ${expected.configDigest.take(n = 12)}...) but was loaded with " +
        "${actual.kind}/${actual.dimension} (digest ${actual.configDigest.take(n = 12)}...); " +
        "scoring with a different featurisation produces confident but wrong labels",
)
