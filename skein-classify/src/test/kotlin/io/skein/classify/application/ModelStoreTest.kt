package io.skein.classify.application

import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SensitivityEnum
import io.skein.classify.domain.TextField
import java.nio.file.Files
import kotlin.io.path.deleteExisting
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ModelStoreTest {

    private val config = HashingConfig(
        key0 = 12345L,
        key1 = 67890L,
        numFeatures = 1 shl 16,
        charNgramMin = 2,
        charNgramMax = 4,
        wordNgramMin = 1,
        wordNgramMax = 3,
    )

    private val schema = Schema.define {
        text(name = "purpose")
        categorical(name = "city")
        numeric(name = "amount")
        identifier(name = "iban")
        label(name = "category")
    }

    private val observations = listOf(
        LabeledFeatures(
            label = Label(value = "housing"),
            features = FeatureVector(indices = intArrayOf(1, 5, 10), values = floatArrayOf(1f, 2f, 3f)),
        ),
        LabeledFeatures(
            label = Label(value = "income"),
            features = FeatureVector(indices = intArrayOf(2, 7), values = floatArrayOf(4f, 5f)),
        ),
    )

    @Test
    internal fun `round-trip preserves all schema field types`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = config,
                observations = observations,
            )
            val loaded = ModelStore.load(path = path)

            assertTrue(actual = loaded.schema.field(name = "purpose") is TextField)
            assertTrue(actual = loaded.schema.field(name = "city") is CategoricalField)
            assertTrue(actual = loaded.schema.field(name = "amount") is NumericField)
            assertTrue(actual = loaded.schema.field(name = "iban") is IdentifierField)
            assertTrue(actual = loaded.schema.field(name = "category") is LabelField)
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `round-trip preserves field sensitivity`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = config,
                observations = emptyList(),
            )
            val loaded = ModelStore.load(path = path)

            assertEquals(
                expected = SensitivityEnum.PUBLIC,
                actual = loaded.schema.field(name = "purpose")?.sensitivity,
            )
            assertEquals(
                expected = SensitivityEnum.PII,
                actual = loaded.schema.field(name = "iban")?.sensitivity,
            )
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `round-trip preserves classifier kind for naive bayes`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = config,
                observations = emptyList(),
            )
            assertEquals(
                expected = ClassifierKindEnum.NAIVE_BAYES,
                actual = ModelStore.load(path = path).classifier,
            )
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `round-trip preserves classifier kind for logistic regression`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.LOGISTIC_REGRESSION,
                hashingConfig = config,
                observations = emptyList(),
            )
            assertEquals(
                expected = ClassifierKindEnum.LOGISTIC_REGRESSION,
                actual = ModelStore.load(path = path).classifier,
            )
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `round-trip preserves all hashing config parameters`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = config,
                observations = emptyList(),
            )
            val loaded = ModelStore.load(path = path)

            assertEquals(expected = config.key0, actual = loaded.hashingConfig.key0)
            assertEquals(expected = config.key1, actual = loaded.hashingConfig.key1)
            assertEquals(expected = config.numFeatures, actual = loaded.hashingConfig.numFeatures)
            assertEquals(expected = config.charNgramMin, actual = loaded.hashingConfig.charNgramMin)
            assertEquals(expected = config.charNgramMax, actual = loaded.hashingConfig.charNgramMax)
            assertEquals(expected = config.wordNgramMin, actual = loaded.hashingConfig.wordNgramMin)
            assertEquals(expected = config.wordNgramMax, actual = loaded.hashingConfig.wordNgramMax)
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `round-trip preserves training observations with labels and feature vectors`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = config,
                observations = observations,
            )
            val loaded = ModelStore.load(path = path)

            assertEquals(expected = 2, actual = loaded.observations.size)
            assertEquals(expected = Label(value = "housing"), actual = loaded.observations[0].label)
            assertEquals(expected = Label(value = "income"), actual = loaded.observations[1].label)
            assertContentEquals(
                expected = intArrayOf(1, 5, 10),
                actual = loaded.observations[0].features.indices,
            )
            assertContentEquals(
                expected = floatArrayOf(1f, 2f, 3f),
                actual = loaded.observations[0].features.values,
            )
            assertContentEquals(expected = intArrayOf(2, 7), actual = loaded.observations[1].features.indices)
            assertContentEquals(expected = floatArrayOf(4f, 5f), actual = loaded.observations[1].features.values)
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `load rejects file with wrong magic bytes`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            path.toFile().writeBytes(byteArrayOf(0x00, 0x00, 0x00, 0x00, 0x01, 0x00))
            assertFailsWith<IllegalArgumentException> {
                ModelStore.load(path = path)
            }
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `load rejects file with correct magic but wrong version`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            // SKEI magic (0x53 0x4B 0x45 0x49) with version 0x02 instead of 0x01
            path.toFile().writeBytes(byteArrayOf(0x53, 0x4B, 0x45, 0x49, 0x02, 0x00))
            assertFailsWith<IllegalArgumentException> {
                ModelStore.load(path = path)
            }
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `load rejects file truncated before the 5-byte header is complete`() {
        val path = Files.createTempFile("skein-test", ".skein")
        try {
            path.toFile().writeBytes(byteArrayOf(0x53, 0x4B, 0x45))
            assertFailsWith<IllegalArgumentException> {
                ModelStore.load(path = path)
            }
        } finally {
            path.deleteExisting()
        }
    }
}
