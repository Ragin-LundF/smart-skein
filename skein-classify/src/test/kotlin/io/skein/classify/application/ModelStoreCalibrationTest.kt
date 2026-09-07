package io.skein.classify.application

import io.skein.classify.domain.Calibration
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Schema
import java.nio.file.Files
import kotlin.io.path.deleteExisting
import kotlin.io.path.readBytes
import kotlin.io.path.writeBytes
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class ModelStoreCalibrationTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }
    private val hashing = HashingConfig(key0 = 4L, key1 = 9L)
    private val observations = listOf(
        LabeledFeatures(
            label = Label(value = "RENT"),
            features = FeatureVector(indices = intArrayOf(1, 2), values = floatArrayOf(1.0f, 2.0f)),
        ),
    )

    private fun saveWith(calibration: Calibration): java.nio.file.Path {
        val path = Files.createTempFile("skein-calibration", ".skein")
        ModelStore.save(
            path = path,
            schema = schema,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = hashing,
            observations = observations,
            calibration = calibration,
        )
        return path
    }

    @Test
    internal fun `the calibration temperature round-trips`() {
        val path = saveWith(calibration = Calibration(temperature = 3.75))
        try {
            assertEquals(
                expected = 3.75,
                actual = ModelStore.load(path = path).calibration.temperature,
                absoluteTolerance = 1e-12,
            )
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `the five-argument save writes an uncalibrated model`() {
        val path = Files.createTempFile("skein-plain", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = hashing,
                observations = observations,
            )
            assertEquals(expected = Calibration.NONE, actual = ModelStore.load(path = path).calibration)
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `a version 1 file still loads and reports no calibration`() {
        // Rewrite a freshly written v2 file's version byte back to v1. The payload is unchanged,
        // which is exactly the guarantee that matters: v2 only APPENDED a field, so a v1 reader's
        // bytes remain a valid prefix and a v2 reader applies the default.
        val path = saveWith(calibration = Calibration.NONE)
        try {
            val bytes = path.readBytes()
            bytes[4] = 0x01
            path.writeBytes(array = bytes)

            val loaded = ModelStore.load(path = path)
            assertEquals(expected = Calibration.NONE, actual = loaded.calibration)
            assertEquals(expected = 1, actual = loaded.observations.size)
            assertEquals(expected = "category", actual = loaded.schema.labelField.name)
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `a truncated payload fails as a malformed file`() {
        val path = saveWith(calibration = Calibration.NONE)
        try {
            val bytes = path.readBytes()
            path.writeBytes(array = bytes.copyOfRange(fromIndex = 0, toIndex = 8))

            val failure = assertFailsWith<IllegalArgumentException> { ModelStore.load(path = path) }
            assertEquals(expected = "truncated .skein file", actual = failure.message)
        } finally {
            path.deleteExisting()
        }
    }
}
