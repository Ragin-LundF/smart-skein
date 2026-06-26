package io.skein.extract.application

import kotlin.test.Test
import kotlin.test.assertEquals

class TemplateClustererTest {

    private val clusterer = TemplateClusterer()

    @Test
    fun `groups texts that share a structural signature`() {
        // Two "<word> <date> <amount>" texts and one "<word> <word>" text.
        val clusters = clusterer.cluster(
            listOf(
                "booked 2024-12-31 12.50",
                "reversal 2025-01-01 99.00",
                "hello world",
            ),
        )
        assertEquals(expected = 2, actual = clusters.size)
        // Largest cluster (the two structured lines) comes first.
        assertEquals(expected = 2, actual = clusters.first().members.size)
        assertEquals(
            expected = listOf("booked 2024-12-31 12.50", "reversal 2025-01-01 99.00"),
            actual = clusters.first().members,
        )
    }

    @Test
    fun `returns one cluster when all texts share a shape`() {
        val clusters = clusterer.cluster(listOf("alpha 1.00", "beta 2.00"))
        assertEquals(expected = 1, actual = clusters.size)
        assertEquals(expected = 2, actual = clusters.first().members.size)
    }
}
