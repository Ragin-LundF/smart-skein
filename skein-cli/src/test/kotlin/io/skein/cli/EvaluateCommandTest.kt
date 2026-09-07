package io.skein.cli

import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.io.path.createTempDirectory
import kotlin.io.path.div
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class EvaluateCommandTest {

    private val workspace = createTempDirectory(prefix = "skein-evaluate")
    private val modelFile = workspace / "model.skein"

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }
    private val hashing = HashingConfig(key0 = 11L, key1 = 22L)

    private val trainingRows = listOf(
        "rent transfer to landlord" to "RENT",
        "monthly rent payment landlord" to "RENT",
        "rent for the apartment" to "RENT",
        "supermarket groceries food" to "FOOD",
        "groceries at the supermarket" to "FOOD",
        "food market groceries" to "FOOD",
    )

    @AfterTest
    fun cleanup() {
        workspace.toFile().deleteRecursively()
    }

    private fun saveModel(): CliEngine {
        val engine = CliEngine.fresh(
            schema = schema,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = hashing,
        )
        trainingRows.forEach { (purpose, category) ->
            engine.service.learn(
                record = Record(values = mapOf("purpose" to purpose, "category" to category)),
            )
        }
        engine.save(path = modelFile)
        return engine
    }

    private fun writeInputCsv(): String {
        val path = workspace / "input.csv"
        val lines = listOf("purpose,category") + trainingRows.map { (purpose, category) -> "$purpose,$category" }
        path.writeText(text = lines.joinToString(separator = "\n", postfix = "\n"))
        return path.toString()
    }

    @Test
    internal fun `rejects mixing input with a retraining mode`() {
        saveModel()
        val failure = assertFailsWith<IllegalArgumentException> {
            runEvaluate(
                flags = mapOf("model" to modelFile.toString(), "input" to writeInputCsv(), "folds" to "5"),
            )
        }
        assertTrue(actual = failure.message.orEmpty().contains(other = "--input"))
    }

    @Test
    internal fun `rejects mixing folds with test-ratio`() {
        saveModel()
        assertFailsWith<IllegalArgumentException> {
            runEvaluate(flags = mapOf("model" to modelFile.toString(), "folds" to "3", "test-ratio" to "0.2"))
        }
    }

    @Test
    internal fun `rejects an unknown flag`() {
        saveModel()
        assertFailsWith<IllegalArgumentException> {
            runEvaluate(flags = mapOf("model" to modelFile.toString(), "foldss" to "3"))
        }
    }

    @Test
    internal fun `scores the saved model against a labeled input csv`() {
        saveModel()
        val reportPath = workspace / "report.txt"
        val status = runEvaluate(
            flags = mapOf(
                "model" to modelFile.toString(),
                "input" to writeInputCsv(),
                "out" to reportPath.toString(),
            ),
        )

        assertEquals(expected = EXIT_OK, actual = status)
        val text = reportPath.readText()
        assertTrue(actual = text.contains(other = "saved model vs"))
        assertTrue(actual = text.contains(other = "accuracy"))
        assertTrue(actual = text.contains(other = "RENT"))
    }

    @Test
    internal fun `rejects an input csv without the label column`() {
        saveModel()
        val path = workspace / "unlabeled.csv"
        path.writeText(text = "purpose\nrent transfer to landlord\n")

        val failure = assertFailsWith<IllegalArgumentException> {
            runEvaluate(flags = mapOf("model" to modelFile.toString(), "input" to path.toString()))
        }
        assertTrue(actual = failure.message.orEmpty().contains(other = "category"))
    }

    @Test
    internal fun `cross-validates the stored corpus and names the mode in the report`() {
        saveModel()
        val reportPath = workspace / "cv.txt"
        val status = runEvaluate(
            flags = mapOf(
                "model" to modelFile.toString(),
                "folds" to "3",
                "out" to reportPath.toString(),
            ),
        )

        assertEquals(expected = EXIT_OK, actual = status)
        val text = reportPath.readText()
        assertTrue(actual = text.contains(other = "3-fold cross-validation"))
        assertTrue(actual = text.contains(other = "retrained per fold"))
        assertTrue(actual = text.contains(other = "per-fold accuracy"))
    }

    @Test
    internal fun `holds out from the stored corpus by default`() {
        saveModel()
        val reportPath = workspace / "holdout.txt"
        runEvaluate(
            flags = mapOf("model" to modelFile.toString(), "out" to reportPath.toString()),
        )

        assertTrue(actual = reportPath.readText().contains(other = "stratified holdout"))
    }

    @Test
    internal fun `writes per-label metrics and the confusion matrix as csv`() {
        saveModel()
        val csvPath = workspace / "per-label.csv"
        val confusionPath = workspace / "confusion.csv"
        runEvaluate(
            flags = mapOf(
                "model" to modelFile.toString(),
                "input" to writeInputCsv(),
                "csv" to csvPath.toString(),
                "confusion" to confusionPath.toString(),
            ),
        )

        val perLabel = CsvCodec.parse(text = csvPath.readText(), delimiter = ',')
        assertEquals(
            expected = listOf("label", "precision", "recall", "f1", "support"),
            actual = perLabel.first(),
        )
        assertEquals(expected = 2, actual = perLabel.drop(n = 1).size)

        val confusion = CsvCodec.parse(text = confusionPath.readText(), delimiter = ',')
        assertEquals(expected = listOf("expected", "FOOD", "RENT"), actual = confusion.first())
        assertEquals(expected = 2, actual = confusion.drop(n = 1).size)
    }

    /** Ground truth deliberately contradicting what the model learned, so accuracy drops below 1. */
    private fun writeMislabeledCsv(): String {
        val path = workspace / "mislabeled.csv"
        val lines = listOf(
            "purpose,category",
            "rent transfer to landlord,FOOD",
            "supermarket groceries food,FOOD",
        )
        path.writeText(text = lines.joinToString(separator = "\n", postfix = "\n"))
        return path.toString()
    }

    @Test
    internal fun `fails the quality gate without suppressing the report`() {
        saveModel()
        val reportPath = workspace / "gated.txt"
        val status = runEvaluate(
            flags = mapOf(
                "model" to modelFile.toString(),
                "input" to writeMislabeledCsv(),
                "out" to reportPath.toString(),
                "min-accuracy" to "1.0",
            ),
        )

        assertEquals(expected = EXIT_QUALITY_GATE_FAILED, actual = status)
        assertTrue(actual = reportPath.readText().contains(other = "accuracy"))
    }

    @Test
    internal fun `passes the quality gate when accuracy clears the threshold`() {
        saveModel()
        val status = runEvaluate(
            flags = mapOf(
                "model" to modelFile.toString(),
                "input" to writeInputCsv(),
                "min-accuracy" to "0.1",
            ),
        )
        assertEquals(expected = EXIT_OK, actual = status)
    }
}
