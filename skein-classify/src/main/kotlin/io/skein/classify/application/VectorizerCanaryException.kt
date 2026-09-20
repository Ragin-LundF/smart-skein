package io.skein.classify.application

import io.skein.classify.domain.VectorizerCanary

/**
 * Thrown when a model's canary probes no longer embed to the vectors they did at training time.
 *
 * **What it means.** The vectorizer's declared identity checked out — same kind, same width, same
 * configuration digest — but it is producing different vectors. For an external embedding service
 * that is exactly the expected shape of the failure: the model was changed on the server, the
 * declared revision was not updated, and nothing in the protocol would have revealed it.
 *
 * **Why this is fatal rather than a warning**, for the same reason as
 * [VectorizerMismatchException]: the model keeps scoring, the labels keep looking plausible, and
 * the damage shows up only as quietly degraded predictions.
 *
 * **This is not thrown when the vectorizer could not be reached.** A timeout or a connection
 * failure propagates as itself, because "your model changed" and "the network is down" call for
 * completely different responses.
 *
 * If the drift is legitimate — a model you deliberately updated — the fix is to retrain and capture
 * a fresh canary, not to widen [VectorizerCanary.tolerance].
 */
class VectorizerCanaryException(
    val canary: VectorizerCanary,
    val probe: String,
    val drift: Double,
) : IllegalStateException(
    "vectorizer canary drifted by $drift, above the tolerance of ${canary.tolerance}: " +
        "the probe \"${probe.take(n = 60)}\" no longer embeds to the vector saved with this model, " +
        "so the featurisation has changed even though its fingerprint has not; " +
        "scoring anyway produces confident but wrong labels",
)
