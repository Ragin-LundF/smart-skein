package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import io.skein.text.application.TypedTokenizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** Snapshot and restore behaviour, isolated from the on-disk codec. */
internal class CrfSnapshotTest {

    private val tokenizer = TypedTokenizer()

    private fun trained(): CrfSequenceLabeler {
        val labeler = CrfSequenceLabeler(initialLearningRate = 0.2, decayRate = 0.05, l2Regularization = 0.01)
        repeat(times = 20) {
            labeler.learn(
                tokens = tokenizer.tokenize(text = "invoice 4711 paid"),
                tags = listOf("KEY", "VALUE", "O").map { tag -> Tag(value = tag) },
            )
        }
        return labeler
    }

    @Test
    internal fun `a snapshot carries the hyperparameters and training progress`() {
        val snapshot = trained().snapshot()

        assertEquals(expected = 0.2, actual = snapshot.initialLearningRate)
        assertEquals(expected = 0.05, actual = snapshot.decayRate)
        assertEquals(expected = 0.01, actual = snapshot.l2Regularization)
        assertEquals(expected = 20L, actual = snapshot.step)
    }

    @Test
    internal fun `an empty sequence is not counted as a training step`() {
        val labeler = trained()
        val before = labeler.snapshot()
        labeler.learn(tokens = emptyList(), tags = emptyList())

        assertEquals(expected = before, actual = labeler.snapshot())
    }

    @Test
    internal fun `a snapshot is a defensive copy`() {
        val labeler = trained()
        val taken = labeler.snapshot()
        labeler.learn(
            tokens = tokenizer.tokenize(text = "receipt 9001 sent"),
            tags = listOf("KEY", "VALUE", "O").map { tag -> Tag(value = tag) },
        )

        assertEquals(expected = 20L, actual = taken.step)
        assertTrue(actual = labeler.snapshot().step > taken.step)
    }

    @Test
    internal fun `restoring in memory reproduces the labeling`() {
        val original = trained()
        val restored = CrfSequenceLabeler.from(snapshot = original.snapshot())
        val probe = tokenizer.tokenize(text = "invoice 5555 paid")

        assertEquals(expected = original.label(tokens = probe), actual = restored.label(tokens = probe))
    }

    @Test
    internal fun `restore rejects an inconsistent snapshot`() {
        val valid = trained().snapshot()

        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler.from(snapshot = valid.copy(tagOrder = emptyList()))
        }
        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler.from(snapshot = valid.copy(tagOrder = valid.tagOrder + valid.tagOrder.first()))
        }
        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler.from(snapshot = valid.copy(step = -1L))
        }
        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler.from(
                snapshot = valid.copy(startWeights = valid.startWeights + (Tag(value = "GHOST") to 1.0)),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler.from(
                snapshot = valid.copy(
                    stateWeights = valid.stateWeights + ((Tag(value = "GHOST") to "type=WORD") to 1.0),
                ),
            )
        }
    }

    @Test
    internal fun `feature counts only tally lexical and structural features seen in training`() {
        val snapshot = trained().snapshot()

        assertTrue(actual = snapshot.featureCounts.keys.any { key -> key.startsWith(prefix = "word=") })
        assertTrue(actual = snapshot.featureCounts.keys.any { key -> key.startsWith(prefix = "type=") })
        assertTrue(actual = snapshot.featureCounts.values.all { count -> count > 0 })
    }
}
