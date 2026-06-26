package io.skein.store.postgres.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.store.postgres.config.JdbcConnectionConfig
import org.testcontainers.containers.PostgreSQLContainer
import org.testcontainers.junit.jupiter.Container
import org.testcontainers.junit.jupiter.Testcontainers
import javax.sql.DataSource
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@Testcontainers
class PostgresFeatureStoreIntegrationTest {

    private fun observation(label: String, vararg indices: Int): LabeledFeatures {
        return LabeledFeatures(
            label = Label(label),
            features = FeatureVector(indices = indices, values = FloatArray(indices.size) { 1.0f }),
        )
    }

    private fun dataSource(): DataSource {
        return TomcatJdbcDataSourceFactory.create(
            JdbcConnectionConfig(
                jdbcUrl = postgres.jdbcUrl,
                username = postgres.username,
                password = postgres.password,
            ),
        )
    }

    @AfterTest
    fun cleanUp() {
        // Each test creates its own store/table; clear shared table to keep tests independent.
        PostgresFeatureStore(dataSource()).clear()
    }

    @Test
    fun `persists and reads back observations preserving order`() {
        val store = PostgresFeatureStore(dataSource())
        store.add(observation("housing", 1, 2, 3))
        store.add(observation("income", 4, 5))

        val all = store.all()
        assertEquals(expected = 2, actual = store.size())
        assertEquals(expected = setOf(Label("housing"), Label("income")), actual = store.labels())
        assertContentEquals(expected = intArrayOf(1, 2, 3), actual = all.first().features.indices)
    }

    @Test
    fun `clear removes all rows`() {
        val store = PostgresFeatureStore(dataSource())
        store.add(observation("a", 1))
        store.clear()
        assertEquals(expected = 0, actual = store.size())
        assertTrue(store.all().isEmpty())
    }

    @Test
    fun `addAll persists every observation in one batch`() {
        val store = PostgresFeatureStore(dataSource())
        store.addAll(
            listOf(
                observation("housing", 1, 2),
                observation("income", 3),
                observation("food", 4, 5, 6),
            ),
        )
        assertEquals(expected = 3, actual = store.size())
        assertEquals(expected = setOf(Label("housing"), Label("income"), Label("food")), actual = store.labels())
    }

    @Test
    fun `encrypted store round-trips while storing ciphertext at rest`() {
        val key = ByteArray(32) { index -> index.toByte() }
        val encryptedStore = PostgresFeatureStore(
            dataSource = dataSource(),
            encryption = AesGcmEncryption(key),
        )
        encryptedStore.add(observation("secret", 7, 8, 9))

        // Reads back correctly through the encrypting store.
        assertContentEquals(expected = intArrayOf(7, 8, 9), actual = encryptedStore.all().first().features.indices)

        // Raw bytes on disk are NOT the plaintext codec output.
        val plaintext = FeatureVectorCodec().encode(observation("secret", 7, 8, 9).features)
        assertFalse(rawStoredBytes().contentEquals(plaintext), message = "stored blob must be encrypted")
    }

    private fun rawStoredBytes(): ByteArray {
        dataSource().connection.use { connection ->
            connection.prepareStatement("SELECT features FROM skein_feature_observation LIMIT 1").use { statement ->
                statement.executeQuery().use { rows ->
                    rows.next()
                    return rows.getBytes("features")
                }
            }
        }
    }

    companion object {
        @Container
        @JvmStatic
        val postgres: PostgreSQLContainer<*> = PostgreSQLContainer("postgres:16-alpine")
    }
}
