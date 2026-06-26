package io.skein.cli

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FlagValidationTest {

    @Test
    fun `accepts a map of only known flags`() {
        requireKnownFlags(
            flags = mapOf("input" to "a.csv", "label-col" to "category", "out" to "b.csv"),
            allowed = LABEL_FLAGS,
        )
    }

    @Test
    fun `rejects a typo'd flag and names it`() {
        val error = assertFailsWith<IllegalArgumentException> {
            requireKnownFlags(flags = mapOf("budgett" to "5"), allowed = LABEL_FLAGS)
        }
        assertTrue(actual = error.message?.contains("--budgett") == true, message = "names the offending flag")
    }

    @Test
    fun `predict rejects a label-only flag`() {
        assertFailsWith<IllegalArgumentException> {
            requireKnownFlags(flags = mapOf("strategy" to "margin"), allowed = PREDICT_FLAGS)
        }
    }
}
