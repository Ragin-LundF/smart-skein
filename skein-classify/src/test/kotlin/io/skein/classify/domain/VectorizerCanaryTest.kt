package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class VectorizerCanaryTest {

    private fun canary(
        probes: List<String> = listOf("alpha", "beta"),
        references: List<FloatArray> = listOf(floatArrayOf(1.0f, 0.0f, 0.0f), floatArrayOf(0.0f, 3.0f, 4.0f)),
        tolerance: Double = VectorizerCanary.DEFAULT_TOLERANCE,
    ): VectorizerCanary {
        return VectorizerCanary(probes = probes, references = references, tolerance = tolerance)
    }

    @Test
    internal fun `identical vectors do not drift`() {
        val subject = canary()

        assertEquals(expected = 0.0, actual = subject.worstDrift(observed = subject.references).distance)
        assertTrue(actual = subject.matches(observed = subject.references))
    }

    @Test
    internal fun `drift is relative to the reference's own magnitude`() {
        // Reference (0,3,4) has norm 5; moving it by 0.05 is a relative drift of 0.01.
        val subject = canary()
        val observed = listOf(floatArrayOf(1.0f, 0.0f, 0.0f), floatArrayOf(0.05f, 3.0f, 4.0f))

        val drift = subject.worstDrift(observed = observed).distance
        assertEquals(expected = 0.01, actual = drift, absoluteTolerance = 1e-9)
    }

    @Test
    internal fun `reports which probe drifted worst`() {
        val subject = canary()
        val observed = listOf(floatArrayOf(1.0f, 0.0f, 0.0f), floatArrayOf(0.0f, 3.0f, 9.0f))

        assertEquals(expected = "beta", actual = subject.worstDrift(observed = observed).probe)
    }

    @Test
    internal fun `noise below the tolerance is accepted`() {
        val subject = canary()
        val observed = listOf(floatArrayOf(1.000001f, 0.0f, 0.0f), floatArrayOf(0.0f, 3.000002f, 4.0f))

        assertTrue(actual = subject.matches(observed = observed))
    }

    /**
     * The test that justifies relative L2 over cosine distance. A service that starts
     * L2-normalising its output rescales every vector and rotates none, so cosine sees nothing —
     * while a linear classifier trained on the unnormalised vectors is broken.
     */
    @Test
    internal fun `a uniform rescaling is caught, which cosine distance would miss`() {
        val subject = canary()
        val observed = subject.references.map { reference ->
            FloatArray(size = reference.size) { index -> reference[index] * 1.5f }
        }

        assertFalse(actual = subject.matches(observed = observed))
        val drift = subject.worstDrift(observed = observed).distance
        assertEquals(expected = 0.5, actual = drift, absoluteTolerance = 1e-6)
    }

    @Test
    internal fun `a change of width is the largest drift there is`() {
        val subject = canary()
        val observed = listOf(floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f), floatArrayOf(0.0f, 3.0f, 4.0f))

        val drift = subject.worstDrift(observed = observed).distance
        assertEquals(expected = VectorizerCanary.MAXIMUM_DRIFT, actual = drift)
    }

    @Test
    internal fun `rejects probes that would make the check meaningless`() {
        assertFailsWith<IllegalArgumentException> { canary(probes = emptyList(), references = emptyList()) }
        assertFailsWith<IllegalArgumentException> {
            canary(probes = listOf("same", "same"), references = listOf(floatArrayOf(1.0f), floatArrayOf(2.0f)))
        }
        assertFailsWith<IllegalArgumentException> {
            canary(probes = listOf(" "), references = listOf(floatArrayOf(1.0f)))
        }
        // Probes ride in the model file in clear text, so a corpus record pasted in is refused.
        assertFailsWith<IllegalArgumentException> {
            canary(probes = listOf("x".repeat(n = 257)), references = listOf(floatArrayOf(1.0f)))
        }
    }

    @Test
    internal fun `rejects references that cannot be compared`() {
        assertFailsWith<IllegalArgumentException> {
            canary(probes = listOf("a", "b"), references = listOf(floatArrayOf(1.0f)))
        }
        assertFailsWith<IllegalArgumentException> {
            canary(probes = listOf("a", "b"), references = listOf(floatArrayOf(1.0f), floatArrayOf(1.0f, 2.0f)))
        }
        // A zero reference has no magnitude to be relative to, so the probe would check nothing.
        assertFailsWith<IllegalArgumentException> {
            canary(probes = listOf("a"), references = listOf(floatArrayOf(0.0f, 0.0f)))
        }
    }

    @Test
    internal fun `rejects a tolerance that would accept anything`() {
        assertFailsWith<IllegalArgumentException> { canary(tolerance = 0.0) }
        assertFailsWith<IllegalArgumentException> { canary(tolerance = -1.0) }
        assertFailsWith<IllegalArgumentException> { canary(tolerance = 0.9) }
    }
}
