package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class GroupedSplitterTest {

    private val splitter = GroupedSplitter()

    private fun observation(group: String?, index: Int): MultiLabeledFeatures {
        return MultiLabeledFeatures(
            features = FeatureVector(indices = intArrayOf(index), values = floatArrayOf(1.0f)),
            labels = setOf(Label(value = "L")),
            group = group,
        )
    }

    /**
     * The property that makes the split honest. If one group straddles two folds, the model sees a
     * sibling of every held-out row during training and the score measures memorisation.
     */
    @Test
    internal fun `never splits a group across folds`() {
        val groups = (0 until 40).map { index -> "rule-${index % 8}" }

        val assignment = splitter.assign(groups = groups, folds = 4)

        groups.toSet().forEach { group ->
            val foldsUsed = groups.indices
                .filter { position -> groups[position] == group }
                .map { position -> assignment[position] }
                .toSet()
            assertEquals(expected = 1, actual = foldsUsed.size, message = "$group spanned $foldsUsed")
        }
    }

    @Test
    internal fun `balances folds despite uneven group sizes`() {
        // Sizes 10, 7, 5, 4, 3, 1: greedy largest-first keeps the folds close despite the spread.
        val groups = buildList {
            repeat(times = 10) { add(element = "a") }
            repeat(times = 7) { add(element = "b") }
            repeat(times = 5) { add(element = "c") }
            repeat(times = 4) { add(element = "d") }
            repeat(times = 3) { add(element = "e") }
            add(element = "f")
        }

        val assignment = splitter.assign(groups = groups, folds = 3)

        val load = IntArray(size = 3)
        assignment.forEach { fold -> load[fold]++ }
        assertEquals(expected = 30, actual = load.sum())
        assertTrue(
            actual = abs(n = load.max() - load.min()) <= 3,
            message = "folds were unbalanced: ${load.toList()}",
        )
    }

    @Test
    internal fun `assigns identically however the corpus is ordered`() {
        val groups = listOf("a", "b", "a", "c", "b", "a", "d", "c")

        val forward = splitter.assign(groups = groups, folds = 2)
        val shuffled = groups.reversed()
        val backward = splitter.assign(groups = shuffled, folds = 2)

        // Compare by group rather than by position, since the positions themselves moved.
        groups.toSet().forEach { group ->
            val fromForward = assignment(groups = groups, assignment = forward, group = group)
            val fromBackward = assignment(groups = shuffled, assignment = backward, group = group)
            assertEquals(expected = fromForward, actual = fromBackward, message = "group $group moved")
        }
    }

    private fun assignment(groups: List<String?>, assignment: IntArray, group: String): Int {
        return assignment[groups.indexOf(element = group)]
    }

    @Test
    internal fun `treats an ungrouped row as a group of one`() {
        val groups = listOf<String?>(null, null, null, null)

        val assignment = splitter.assign(groups = groups, folds = 2)

        val load = IntArray(size = 2)
        assignment.forEach { fold -> load[fold]++ }
        assertEquals(expected = listOf(2, 2), actual = load.toList())
    }

    @Test
    internal fun `mixes grouped and ungrouped rows without collision`() {
        // A caller-supplied group named like the internal singleton key must stay distinct.
        val groups = listOf("a", "a", null, null, "b", "b")

        val assignment = splitter.assign(groups = groups, folds = 2)

        assertEquals(expected = assignment[0], actual = assignment[1])
        assertEquals(expected = assignment[4], actual = assignment[5])
    }

    @Test
    internal fun `holds out each observation exactly once across the folds`() {
        val observations = (0 until 12).map { index -> observation(group = "g${index % 4}", index = index) }

        val splits = splitter.folds(observations = observations, folds = 4)

        assertEquals(expected = 4, actual = splits.size)
        assertEquals(expected = 12, actual = splits.sumOf { split -> split.holdout.size })
        splits.forEach { split ->
            assertEquals(expected = 12, actual = split.training.size + split.holdout.size)
            assertTrue(actual = split.holdout.none { row -> row in split.training })
        }
    }

    /**
     * The regression this ordering exists for. Group names encode structure, so assigning
     * equal-sized groups in plain alphabetical order can line a fold up with a whole category: five
     * methods cycling under five folds put every `one pot` group in one fold, and the label that
     * depends on that method was then missing from four folds' training data entirely.
     */
    @Test
    internal fun `does not line a fold up with a category encoded in the group names`() {
        val methods = listOf("baked", "fried", "grilled", "one pot", "simmered")
        val groups = buildList {
            listOf("italian", "mexican", "plain", "sweet").forEach { cuisine ->
                listOf("beef", "chicken", "pork", "fish", "tofu", "beans").forEach { protein ->
                    methods.forEach { method ->
                        repeat(times = 2) { add(element = "$cuisine/$protein/$method") }
                    }
                }
            }
        }

        val assignment = splitter.assign(groups = groups, folds = methods.size)

        methods.forEach { method ->
            val folds = groups.indices
                .filter { position -> groups[position].endsWith(suffix = "/$method") }
                .map { position -> assignment[position] }
                .toSet()
            assertTrue(
                actual = folds.size > 1,
                message = "every '$method' group landed in the same fold: $folds",
            )
        }
    }

    @Test
    internal fun `a different seed produces a different assignment`() {
        val groups = (0 until 40).map { index -> "group-${index / 2}" }

        val first = GroupedSplitter(seed = 1L).assign(groups = groups, folds = 4)
        val second = GroupedSplitter(seed = 2L).assign(groups = groups, folds = 4)

        assertTrue(actual = !first.contentEquals(other = second))
    }

    @Test
    internal fun `the same seed reproduces the same assignment`() {
        val groups = (0 until 40).map { index -> "group-${index / 2}" }

        assertTrue(
            actual = GroupedSplitter(seed = 7L).assign(groups = groups, folds = 4)
                .contentEquals(other = GroupedSplitter(seed = 7L).assign(groups = groups, folds = 4)),
        )
    }

    @Test
    internal fun `rejects a corpus or fold count it cannot split`() {
        assertFailsWith<IllegalArgumentException> { splitter.assign(groups = emptyList(), folds = 2) }
        assertFailsWith<IllegalArgumentException> { splitter.assign(groups = listOf("a"), folds = 1) }
        assertFailsWith<IllegalArgumentException> { splitter.assign(groups = listOf("a", "b"), folds = 5) }
    }
}
