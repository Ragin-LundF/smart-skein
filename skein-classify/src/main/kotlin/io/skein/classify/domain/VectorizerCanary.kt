package io.skein.classify.domain

import kotlin.math.sqrt

/**
 * A handful of fixed texts and the vectors a [io.skein.classify.spi.Vectorizer] produced for them
 * at training time, kept so a featurisation that changes underneath a saved model is caught.
 *
 * **What this catches that [VectorizerFingerprint] cannot.** A fingerprint covers what the client
 * can see. That is everything for feature hashing, whose output is fully determined by its
 * configuration, and everything for a local ONNX model, whose file bytes are hashed. It is not
 * enough for an external embedding service: it returns vectors, and nothing in the protocol says
 * which weights produced them. A model updated on the server keeps the same name, the same declared
 * revision and the same width, so the fingerprint still matches — and the vectors are different.
 *
 * Re-embedding these probes on load turns that silent failure into a loud one. It is the only check
 * that looks at what a vectorizer actually *does* rather than at what it claims to be.
 *
 * ## Choosing probes
 *
 * **Never sample them from your corpus.** They are stored in the model file as plain text, which
 * makes them the one thing in a `.skein` file that is not an irreversible hash. Use short, fixed,
 * synthetic sentences you wrote for the purpose. A few is plenty — every probe is re-embedded on
 * every load, and for a remote service that is a network round trip each.
 *
 * Spread them across the languages you care about. A model revision that shifts only one language's
 * subspace is invisible to a probe set that is entirely English.
 *
 * ## The metric
 *
 * Relative L2 distance, `‖observed − reference‖ / ‖reference‖`, rather than cosine distance.
 * Cosine measures direction alone, and a service that starts L2-normalising its output — a toggle
 * in LM Studio and in Text Embeddings Inference — rescales every vector while leaving direction
 * untouched. For a linear classifier trained on unnormalised vectors that is a model-breaking
 * change, and cosine would wave it through. Relative L2 catches rotation and rescaling both, and
 * being dimensionless it means the same thing at 384 and at 1024 dimensions.
 *
 * **Why a tolerance rather than equality.** Floating-point accumulation order varies with batch
 * size and with however many threads a runtime picks, so identical weights do not produce identical
 * bits. That noise lands around `1e-5`; a requantised or retrained model moves the vector by `1e-2`
 * or more. [DEFAULT_TOLERANCE] sits in the gap.
 *
 * @property probes the texts to re-embed. Stored in the model file as plain text.
 * @property references the vectors [probes] produced at training time, aligned by position.
 * @property tolerance the largest relative L2 distance still treated as the same model.
 */
class VectorizerCanary(
    val probes: List<String>,
    val references: List<FloatArray>,
    val tolerance: Double = DEFAULT_TOLERANCE,
) {

    init {
        require(value = probes.isNotEmpty()) { "a canary needs at least one probe text" }
        require(value = probes.none { probe -> probe.isBlank() }) { "a probe text must not be blank" }
        require(value = probes.distinct().size == probes.size) { "probe texts must be distinct" }
        // A nudge, not a guarantee: probes are stored in clear text, and a long one is usually a
        // record someone pasted in rather than a sentence they wrote for the purpose.
        require(value = probes.all { probe -> probe.length <= MAXIMUM_PROBE_LENGTH }) {
            "a probe must be at most $MAXIMUM_PROBE_LENGTH characters; probes are stored in the " +
                "model file in clear text, so they must be short synthetic sentences and never " +
                "records from your corpus"
        }
        require(value = probes.size == references.size) {
            "probes and references must be aligned, got ${probes.size} probes and ${references.size} references"
        }
        require(value = references.all { reference -> reference.isNotEmpty() }) {
            "a reference vector must not be empty"
        }
        require(value = references.distinctBy { reference -> reference.size }.size == 1) {
            "every reference vector must have the same width"
        }
        require(value = references.none { reference -> norm(vector = reference) == 0.0 }) {
            "a reference vector must not be all zeros: relative distance against it is undefined, " +
                "so the probe would check nothing"
        }
        require(value = tolerance > 0.0 && tolerance <= MAXIMUM_TOLERANCE) {
            "tolerance must be in (0, $MAXIMUM_TOLERANCE], got $tolerance"
        }
    }

    /** Width of the reference vectors. */
    fun dimension(): Int {
        return references.first().size
    }

    /**
     * Relative L2 distance between [observed] and the reference for the probe at [index].
     *
     * A width change is reported as [MAXIMUM_DRIFT] rather than as an error, so a caller stays on
     * one failure path instead of two.
     */
    fun drift(index: Int, observed: FloatArray): Double {
        val reference = references[index]
        if (observed.size != reference.size) {
            return MAXIMUM_DRIFT
        }
        var squared = 0.0
        for (position in reference.indices) {
            val delta = observed[position].toDouble() - reference[position].toDouble()
            squared += delta * delta
        }
        return sqrt(x = squared) / norm(vector = reference)
    }

    /**
     * The worst drift across every probe, which is the number [tolerance] is compared against,
     * together with the probe that produced it.
     */
    fun worstDrift(observed: List<FloatArray>): CanaryDrift {
        require(value = observed.size == references.size) {
            "expected ${references.size} vectors to check, got ${observed.size}"
        }
        return references.indices
            .map { index ->
                CanaryDrift(probe = probes[index], distance = drift(index = index, observed = observed[index]))
            }
            .maxBy { drift -> drift.distance }
    }

    /** Whether every probe in [observed] is within [tolerance]. */
    fun matches(observed: List<FloatArray>): Boolean {
        return worstDrift(observed = observed).distance <= tolerance
    }

    private fun norm(vector: FloatArray): Double {
        var squared = 0.0
        for (component in vector) {
            squared += component.toDouble() * component.toDouble()
        }
        return sqrt(x = squared)
    }

    /** The worst-drifting probe and how far it moved. */
    class CanaryDrift(val probe: String, val distance: Double)

    companion object {
        /**
         * One percent. Roughly a thousand times the drift that accumulation order alone produces,
         * and comfortably below what a requantised or retrained model produces.
         *
         * A model requantised from `fp16` to `q8_0` typically moves by two to three percent and so
         * trips this. That is a **true** positive: a requantised model is a different model as far
         * as a classifier trained on its vectors is concerned. Retrain rather than widening this.
         */
        const val DEFAULT_TOLERANCE = 0.01

        /** Beyond this a canary would accept anything, which is worse than having none. */
        const val MAXIMUM_TOLERANCE = 0.5

        /** Reported when the width itself changed, which is as different as two vectors get. */
        const val MAXIMUM_DRIFT = Double.MAX_VALUE

        private const val MAXIMUM_PROBE_LENGTH = 256
    }
}
