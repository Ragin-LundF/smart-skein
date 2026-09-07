package io.skein.cli

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class FlagValidationTest {

    @Test
    internal fun `accepts a map of only known flags`() {
        requireKnownFlags(
            flags = mapOf("input" to "a.csv", "label-col" to "category", "out" to "b.csv"),
            allowed = LABEL_FLAGS,
        )
    }

    @Test
    internal fun `rejects a typo'd flag and names it`() {
        val error = assertFailsWith<IllegalArgumentException> {
            requireKnownFlags(flags = mapOf("budgett" to "5"), allowed = LABEL_FLAGS)
        }
        assertEquals(
            expected = error.message?.contains("--budgett"),
            actual = true,
            message = "names the offending flag"
        )
    }

    @Test
    internal fun `predict rejects a label-only flag`() {
        assertFailsWith<IllegalArgumentException> {
            requireKnownFlags(flags = mapOf("strategy" to "margin"), allowed = PREDICT_FLAGS)
        }
    }

    @Test
    internal fun `evaluate flags accept their known set and reject a typo`() {
        requireKnownFlags(flags = mapOf("model" to "m", "folds" to "5"), allowed = EVALUATE_FLAGS)
        requireKnownFlags(flags = mapOf("min-accuracy" to "0.8", "confusion" to "c.csv"), allowed = EVALUATE_FLAGS)
        assertFailsWith<IllegalArgumentException> {
            requireKnownFlags(flags = mapOf("foldss" to "5"), allowed = EVALUATE_FLAGS)
        }
        assertFailsWith<IllegalArgumentException> {
            requireKnownFlags(flags = mapOf("strategy" to "margin"), allowed = EVALUATE_FLAGS)
        }
    }

    @Test
    internal fun `parseRatio accepts a proper fraction and rejects the boundaries`() {
        assertEquals(expected = 0.25, actual = parseRatio(name = "test-ratio", value = "0.25", default = 0.2))
        assertEquals(expected = 0.2, actual = parseRatio(name = "test-ratio", value = null, default = 0.2))
        listOf("0.0", "1.0", "-0.1", "1.5", "abc").forEach { bad ->
            assertFailsWith<IllegalArgumentException> {
                parseRatio(name = "test-ratio", value = bad, default = 0.2)
            }
        }
    }

    @Test
    internal fun `parsePositiveInt requires at least one`() {
        assertEquals(expected = 7, actual = parsePositiveInt(name = "folds", value = "7", default = 5))
        assertEquals(expected = 5, actual = parsePositiveInt(name = "folds", value = null, default = 5))
        listOf("0", "-3", "x").forEach { bad ->
            assertFailsWith<IllegalArgumentException> { parsePositiveInt(name = "folds", value = bad, default = 5) }
        }
    }

    @Test
    internal fun `parseAccuracyGate spans the inclusive unit interval`() {
        assertEquals(expected = null, actual = parseAccuracyGate(value = null))
        assertEquals(expected = 0.0, actual = parseAccuracyGate(value = "0.0"))
        assertEquals(expected = 1.0, actual = parseAccuracyGate(value = "1.0"))
        listOf("-0.1", "1.1", "abc").forEach { bad ->
            assertFailsWith<IllegalArgumentException> { parseAccuracyGate(value = bad) }
        }
    }
}
