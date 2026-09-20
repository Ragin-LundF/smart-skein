package io.skein.examples.localai

import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.spi.Vectorizer

/**
 * One drift measurement and what it means against a tolerance.
 *
 * @property probe the worst-drifting probe text.
 * @property drift relative L2 distance — `‖observed − reference‖ / ‖reference‖`.
 * @property tolerance the value it was compared against.
 */
data class DriftReading(val probe: String, val drift: Double, val tolerance: Double) {

    /** Whether a load would be refused. */
    val trips: Boolean get() = drift > tolerance

    /** How much room is left before a false positive. Large is good. */
    val headroom: Double get() = tolerance / maxOf(drift, FLOOR)

    private companion object {
        const val FLOOR = 1e-12
    }
}

/**
 * Measures how far an embedding model's output actually moves, so a
 * [VectorizerCanary]'s tolerance can be chosen from data rather than guessed.
 *
 * ## The question this answers
 *
 * A canary refuses to load a model when its probe vectors have moved past a tolerance. Set that
 * tolerance too tight and every load fails on ordinary numerical noise; too loose and a genuinely
 * changed model slips through. The gap between those is usually wide — noise sits far below a real
 * weight change — but *where* it sits depends on the model, the runtime and the hardware, which is
 * exactly what you cannot look up.
 *
 * So measure it. Both measurements are cheap and both are worth doing before you rely on a canary
 * in production:
 *
 * | Measure | With | Expect |
 * |---|---|---|
 * | [noiseFloor] | nothing changed | far below the tolerance |
 * | [driftAgainst] | the model deliberately changed | above the tolerance |
 *
 * ## Interpreting the result
 *
 * - **Noise floor near zero** is normal and the common case: a deterministic runtime asked for one
 *   text at a time returns identical vectors, so there is nothing to drift. `VectorizerCanary`
 *   embeds one probe per request for exactly this reason — batch composition perturbs
 *   floating-point accumulation order, and a reference captured in a batch of thirty-two compared
 *   against one re-embedded alone would drift for a reason that has nothing to do with the model.
 * - **Noise floor above the tolerance** means the default will fire on unchanged models. Raise the
 *   tolerance above what you measured, with room to spare.
 * - **Less than roughly 10x headroom** is uncomfortably close. Expect occasional false positives.
 * - **A deliberately changed model that does *not* trip** means the tolerance is too loose. Lower
 *   it until it does, then confirm the noise floor still clears.
 *
 * If a canary fires and the change was intended, retrain and capture a fresh one. Do not widen the
 * tolerance to silence it — that is turning off the smoke alarm.
 */
class CanaryCalibrator(private val tolerance: Double = VectorizerCanary.DEFAULT_TOLERANCE) {

    /**
     * Captures probes through [vectorizer], then immediately re-embeds them through it to see how
     * far they move when nothing has changed.
     *
     * Pass a *second, independently constructed* vectorizer as [reader] to include connection and
     * batching differences in the measurement — the sources of noise that a single cached client
     * would hide.
     */
    fun noiseFloor(
        vectorizer: Vectorizer,
        probes: List<String>,
        reader: Vectorizer = vectorizer,
    ): DriftReading {
        val captured = VectorizerCanary(
            probes = probes,
            references = probes.map { probe -> vectorizer.vectorize(text = probe).values },
            tolerance = tolerance,
        )
        return driftAgainst(canary = captured, vectorizer = reader)
    }

    /**
     * How far [vectorizer] has moved from what [canary] recorded.
     *
     * This is what `ModelStore.loadMultiLabel` does internally, minus the throwing — use it to
     * *report* drift, on a schedule or at startup, without failing the caller. A long-running
     * service can hold `LoadedMultiLabelModel.canary` and re-check periodically, since a service
     * can be updated at any point and not only while nobody has the model open.
     */
    fun driftAgainst(canary: VectorizerCanary, vectorizer: Vectorizer): DriftReading {
        val observed = canary.probes.map { probe -> vectorizer.vectorize(text = probe).values }
        val worst = canary.worstDrift(observed = observed)
        return DriftReading(probe = worst.probe, drift = worst.distance, tolerance = canary.tolerance)
    }

    /** A one-line verdict suitable for a log line or a startup banner. */
    fun verdict(reading: DriftReading): String {
        val measured = "${format(reading.drift)} vs ${reading.tolerance}"
        return when {
            reading.trips ->
                "DRIFTED ($measured) — the served model is not the one this was trained on"
            reading.headroom < COMFORTABLE_HEADROOM ->
                "close ($measured) — only ${"%.1f".format(reading.headroom)}x headroom, expect false positives"
            else ->
                "unchanged ($measured) — ${"%.0f".format(reading.headroom)}x headroom"
        }
    }

    private fun format(value: Double): String {
        return "%.9f".format(value)
    }

    private companion object {
        const val COMFORTABLE_HEADROOM = 10.0
    }
}
