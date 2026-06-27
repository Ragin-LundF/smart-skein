package io.skein.cli

import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.Record
import io.skein.classify.domain.UncertaintyStrategyEnum
import java.io.BufferedReader
import java.util.Locale
import kotlin.random.Random

private const val ALTERNATIVES_SHOWN = 3

/**
 * Interactive active-learning loop. Trains the engine on the rows that already carry a label, then
 * repeatedly surfaces the rows the model is least sure about ([PoolSelector]) for a human to confirm
 * or correct in the terminal, learning from each answer.
 *
 * I/O is injected ([input]/[output]) so the loop is unit-testable with scripted input. Rows are
 * mutated in place — a freshly labeled row gets its [labelColumn] filled — and the engine learns
 * each answer, so the caller can write the enriched dataset and save the model afterwards.
 *
 * Scales to large pools: feature vectors are cached per row (vectorized once), scored in parallel,
 * and ranked with a bounded heap; [scanLimit] caps the per-round scan via sampling so millions of
 * candidates stay tractable (see [PoolSelector]). The remaining ceiling is that all rows are held in
 * memory at once — stream the input if a pool ever outgrows the heap.
 */
class LabelingSession(
    private val engine: CliEngine,
    private val budget: Int,
    private val batchSize: Int,
    private val strategy: UncertaintyStrategyEnum,
    private val epochs: Int,
    private val scanLimit: Int,
    private val random: Random,
    private val input: BufferedReader,
    private val output: Appendable,
) {

    private val selector = PoolSelector(engine = engine, strategy = strategy, scanLimit = scanLimit, random = random)
    private val labelColumn = engine.labelColumn

    /** Runs the loop over [rows] (mutated in place). Returns how many rows the human labeled. */
    fun run(rows: List<MutableMap<String, Any?>>): Int {
        val labeled = rows.filter { row -> hasLabel(row = row) }
        if (!engine.isTrained()) {
            trainOn(labeled = labeled)
        }
        val pending = rows.asSequence()
            .filter { row -> !hasLabel(row = row) }
            .mapTo(destination = ArrayList()) { row -> PoolEntry(row = row) }
        if (pending.isEmpty()) {
            output.appendLine("No unlabeled rows to review.")
            return 0
        }
        check(engine.isTrained()) {
            "no labeled rows to learn from — give at least one row a value in column '$labelColumn', " +
                "or pass --model with an already-trained model to seed the suggestions"
        }
        return label(pending = pending)
    }

    private fun trainOn(labeled: List<Map<String, Any?>>) {
        if (labeled.isEmpty()) {
            return
        }
        engine.service.learnAll(records = labeled.map { row -> Record(values = row) })
        // SGD needs several passes to converge; Naive Bayes is exact after one and needs no retrain.
        if (engine.classifier == ClassifierKindEnum.LOGISTIC_REGRESSION) {
            engine.service.retrain(epochs = epochs)
        }
    }

    private fun label(pending: MutableList<PoolEntry>): Int {
        var labeledCount = 0
        while (labeledCount < budget && pending.isNotEmpty()) {
            val take = minOf(batchSize, budget - labeledCount)
            val picks = selector.selectMostUncertain(entries = pending, limit = take)
            if (picks.isEmpty()) {
                break
            }
            for (pick in picks) {
                if (labeledCount >= budget) {
                    break
                }
                present(row = pick.row, prediction = pick.prediction)
                when (val reply = readReply(suggested = pick.prediction.label.value)) {
                    Reply.Quit -> return labeledCount
                    Reply.Skip -> pending.remove(element = pick)
                    is Reply.Assign -> {
                        pick.row[labelColumn] = reply.label
                        engine.service.feedback(
                            record = Record(values = pick.row),
                            correctLabel = Label(value = reply.label),
                        )
                        pending.remove(element = pick)
                        labeledCount += 1
                    }
                }
            }
        }
        return labeledCount
    }

    private fun present(row: Map<String, Any?>, prediction: Prediction) {
        output.appendLine("──────────────────────────────")
        row.forEach { (name, value) ->
            if (name != labelColumn) {
                output.appendLine("  $name: $value")
            }
        }
        val confidence = "%.2f".format(Locale.ROOT, prediction.confidence)
        output.appendLine("  suggested: ${prediction.label.value} (confidence $confidence)")
        val others = prediction.alternatives.take(n = ALTERNATIVES_SHOWN).joinToString(separator = ", ") { scored ->
            "${scored.label.value} ${"%.2f".format(Locale.ROOT, scored.probability)}"
        }
        if (others.isNotEmpty()) {
            output.appendLine("  ranked: $others")
        }
        output.append("  label [Enter=accept, <text>=set, s=skip, q=quit]: ")
    }

    private fun readReply(suggested: String): Reply {
        val line = input.readLine() ?: return Reply.Quit
        return when (val answer = line.trim()) {
            "", "y" -> Reply.Assign(label = suggested)
            "q" -> Reply.Quit
            "s" -> Reply.Skip
            else -> Reply.Assign(label = answer)
        }
    }

    private fun hasLabel(row: Map<String, Any?>): Boolean {
        return row[labelColumn]?.toString()?.isNotBlank() == true
    }

    private sealed interface Reply {
        data class Assign(val label: String) : Reply
        data object Skip : Reply
        data object Quit : Reply
    }
}
