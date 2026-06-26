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
}
