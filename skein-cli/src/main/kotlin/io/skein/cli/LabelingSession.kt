package io.skein.cli

import io.skein.classify.application.ActiveLearningSelector
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.Record
import io.skein.classify.domain.UncertaintyStrategyEnum
import java.io.BufferedReader
import java.util.Locale

private const val ALTERNATIVES_SHOWN = 3

/**
 * Interactive active-learning loop. Trains the engine on the rows that already carry a label, then
 * repeatedly surfaces the rows the model is least sure about ([ActiveLearningSelector]) for a human
 * to confirm or correct in the terminal, learning from each answer.
 *
 * I/O is injected ([input]/[output]) so the loop is unit-testable with scripted input. Rows are
 * mutated in place — a freshly labeled row gets its [labelColumn] filled — and the engine learns
 * each answer, so the caller can write the enriched dataset and save the model afterwards.
 *
 * The candidate pool is re-selected after every answer so uncertainty reflects the latest learning.
 * ponytail: re-classifies the whole remaining pool each round — fine at human labeling scale.
 */
class LabelingSession(
    private val engine: CliEngine,
    private val budget: Int,
    private val batchSize: Int,
    private val strategy: UncertaintyStrategyEnum,
    private val epochs: Int,
    private val input: BufferedReader,
    private val output: Appendable,
) {

    private val selector = ActiveLearningSelector(service = engine.service)
    private val labelColumn = engine.labelColumn

    /** Runs the loop over [rows] (mutated in place). Returns how many rows the human labeled. */
    fun run(rows: List<MutableMap<String, Any?>>): Int {
        val labeled = rows.filter { row -> hasLabel(row = row) }
        val unlabeled = rows.filterTo(destination = ArrayList()) { row -> !hasLabel(row = row) }
        trainOn(labeled = labeled)
        if (unlabeled.isEmpty()) {
            output.appendLine("No unlabeled rows to review.")
            return 0
        }
        return label(pending = unlabeled)
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

    private fun label(pending: MutableList<MutableMap<String, Any?>>): Int {
        var labeledCount = 0
        while (labeledCount < budget && pending.isNotEmpty()) {
            val take = minOf(batchSize, budget - labeledCount)
            val candidates = selector.selectForReview(
                candidates = pending.map { row -> Record(values = row) },
                limit = take,
                strategy = strategy,
            )
            if (candidates.isEmpty()) {
                break
            }
            for (candidate in candidates) {
                if (labeledCount >= budget) {
                    break
                }
                val row = pending.first { row -> row === candidate.record.values }
                present(row = row, prediction = candidate.prediction)
                when (val reply = readReply(suggested = candidate.prediction.label.value)) {
                    Reply.Quit -> return labeledCount
                    Reply.Skip -> removeByIdentity(pending = pending, row = row)
                    is Reply.Assign -> {
                        row[labelColumn] = reply.label
                        engine.service.feedback(
                            record = Record(values = row),
                            correctLabel = Label(value = reply.label),
                        )
                        removeByIdentity(pending = pending, row = row)
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

    private fun removeByIdentity(pending: MutableList<MutableMap<String, Any?>>, row: Map<String, Any?>) {
        val index = pending.indexOfFirst { candidate -> candidate === row }
        if (index >= 0) {
            pending.removeAt(index = index)
        }
    }

    private sealed interface Reply {
        data class Assign(val label: String) : Reply
        data object Skip : Reply
        data object Quit : Reply
    }
}
