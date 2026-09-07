package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class StratifiedSplitterTest {

    private var nextIndex = 0

    /** Distinct feature vectors so observations are distinguishable by identity. */
    private fun observation(label: String): LabeledFeatures {
        nextIndex += 1
        return LabeledFeatures(
            label = Label(value = label),
            features = FeatureVector(indices = intArrayOf(nextIndex), values = floatArrayOf(1.0f)),
        )
    }

    private fun corpus(vararg counts: Pair<String, Int>): List<LabeledFeatures> {
        return counts.flatMap { (label, count) -> List(size = count) { observation(label = label) } }
    }

    @Test
    internal fun `holdout preserves each label's proportion`() {
        val observations = corpus("A" to 50, "B" to 30, "C" to 20)
        val split = StratifiedSplitter().holdout(observations = observations, testRatio = 0.2)

        assertEquals(expected = 20, actual = split.holdout.size)
        assertEquals(expected = 80, actual = split.training.size)
        assertEquals(expected = 10, actual = split.holdout.count { it.label == Label(value = "A") })
        assertEquals(expected = 6, actual = split.holdout.count { it.label == Label(value = "B") })
        assertEquals(expected = 4, actual = split.holdout.count { it.label == Label(value = "C") })
    }

    @Test
    internal fun `holdout is a partition of the input`() {
        val observations = corpus("A" to 17, "B" to 9)
        val split = StratifiedSplitter().holdout(observations = observations)

        assertEquals(expected = observations.size, actual = split.training.size + split.holdout.size)
        assertTrue(actual = split.training.intersect(split.holdout.toSet()).isEmpty())
        assertEquals(expected = observations.toSet(), actual = (split.training + split.holdout).toSet())
    }

    @Test
    internal fun `the same seed reproduces a split and a different seed changes it`() {
        val observations = corpus("A" to 40, "B" to 40)
        val first = StratifiedSplitter(seed = 7L).holdout(observations = observations)
        val second = StratifiedSplitter(seed = 7L).holdout(observations = observations)
        val other = StratifiedSplitter(seed = 8L).holdout(observations = observations)

        assertEquals(expected = first.holdout, actual = second.holdout)
        assertTrue(actual = first.holdout != other.holdout)
    }

    @Test
    internal fun `every label keeps at least one training example`() {
        // At ratio 0.9 an unclamped split would move both members of B into the holdout.
        val observations = corpus("A" to 100, "B" to 2)
        val split = StratifiedSplitter().holdout(observations = observations, testRatio = 0.9)

        assertTrue(actual = split.training.any { it.label == Label(value = "B") })
    }

    @Test
    internal fun `a singleton label stays entirely in training`() {
        val observations = corpus("A" to 20, "rare" to 1)
        val split = StratifiedSplitter().holdout(observations = observations, testRatio = 0.5)

        assertEquals(expected = 0, actual = split.holdout.count { it.label == Label(value = "rare") })
        assertEquals(expected = 1, actual = split.training.count { it.label == Label(value = "rare") })
    }

    @Test
    internal fun `training order is not blocked by label`() {
        // Grouping produces all of A then all of B, which is pathological for SGD. The splitter
        // must reshuffle the assembled training list.
        val observations = corpus("A" to 50, "B" to 50)
        val split = StratifiedSplitter().holdout(observations = observations)
        val labels = split.training.map { it.label.value }

        assertTrue(actual = labels != labels.sorted(), message = "training set is still label-blocked")
    }

    @Test
    internal fun `k-fold holds out every observation exactly once`() {
        val observations = corpus("A" to 25, "B" to 15)
        val splits = StratifiedSplitter().folds(observations = observations, folds = 5)

        assertEquals(expected = 5, actual = splits.size)
        val allHoldouts = splits.flatMap { split -> split.holdout }
        assertEquals(expected = observations.size, actual = allHoldouts.size)
        assertEquals(expected = observations.toSet(), actual = allHoldouts.toSet())
    }

    @Test
    internal fun `k-fold sizes differ by at most one and training excludes the fold`() {
        val observations = corpus("A" to 23, "B" to 11)
        val splits = StratifiedSplitter().folds(observations = observations, folds = 4)
        val sizes = splits.map { split -> split.holdout.size }

        assertTrue(actual = sizes.max() - sizes.min() <= 1, message = "fold sizes were $sizes")
        splits.forEach { split ->
            assertTrue(actual = split.training.intersect(split.holdout.toSet()).isEmpty())
            assertEquals(expected = observations.size, actual = split.training.size + split.holdout.size)
        }
    }

    @Test
    internal fun `k-fold preserves label proportions in every fold`() {
        val observations = corpus("A" to 40, "B" to 20)
        val splits = StratifiedSplitter().folds(observations = observations, folds = 4)

        splits.forEach { split ->
            assertEquals(expected = 10, actual = split.holdout.count { it.label == Label(value = "A") })
            assertEquals(expected = 5, actual = split.holdout.count { it.label == Label(value = "B") })
        }
    }

    @Test
    internal fun `rejects invalid split parameters`() {
        val observations = corpus("A" to 4)
        assertFailsWith<IllegalArgumentException> { StratifiedSplitter().holdout(observations = emptyList()) }
        assertFailsWith<IllegalArgumentException> {
            StratifiedSplitter().holdout(observations = observations, testRatio = 0.0)
        }
        assertFailsWith<IllegalArgumentException> {
            StratifiedSplitter().holdout(observations = observations, testRatio = 1.0)
        }
        assertFailsWith<IllegalArgumentException> { StratifiedSplitter().folds(observations = observations, folds = 1) }
        assertFailsWith<IllegalArgumentException> { StratifiedSplitter().folds(observations = observations, folds = 5) }
    }
}
