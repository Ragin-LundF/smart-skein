package io.skein.store.postgres.infrastructure

import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.spi.FeatureStore
import io.skein.store.postgres.spi.FeatureEncryption
import java.sql.SQLException
import javax.sql.DataSource

/**
 * PostgreSQL-backed [FeatureStore]. Feature vectors are serialized with [FeatureVectorCodec] and
 * optionally encrypted at rest via [FeatureEncryption] (use [AesGcmEncryption] for
 * [io.skein.classify.domain.PrivacyModeEnum.ENCRYPTED_SOURCE], [NoEncryption] otherwise).
 *
 * The schema is created on construction via Liquibase ([SchemaMigrator]); the table name is fixed.
 * All values are bound as parameters.
 */
class PostgresFeatureStore(
    private val dataSource: DataSource,
    private val codec: FeatureVectorCodec = FeatureVectorCodec(),
    private val encryption: FeatureEncryption = NoEncryption(),
) : FeatureStore {

    init {
        SchemaMigrator(dataSource = dataSource).migrate()
    }

    override fun add(observation: LabeledFeatures) {
        val stored = encryption.encrypt(codec.encode(observation.features))
        dataSource.connection.use { connection ->
            connection.prepareStatement(INSERT_SQL).use { statement ->
                statement.setString(1, observation.label.value)
                statement.setBytes(2, stored)
                statement.executeUpdate()
            }
        }
    }

    /** Batched insert: all rows go in one prepared statement and one transaction. */
    override fun addAll(observations: Collection<LabeledFeatures>) {
        if (observations.isEmpty()) {
            return
        }
        // Encode (and encrypt) outside the transaction so only JDBC can fail inside it.
        val rows = observations.map { observation ->
            observation.label.value to encryption.encrypt(codec.encode(observation.features))
        }
        dataSource.connection.use { connection ->
            connection.autoCommit = false
            try {
                connection.prepareStatement(INSERT_SQL).use { statement ->
                    for ((label, blob) in rows) {
                        statement.setString(1, label)
                        statement.setBytes(2, blob)
                        statement.addBatch()
                    }
                    statement.executeBatch()
                }
                connection.commit()
            } catch (exception: SQLException) {
                connection.rollback()
                throw exception
            } finally {
                connection.autoCommit = true
            }
        }
    }

    override fun all(): List<LabeledFeatures> {
        val observations = ArrayList<LabeledFeatures>()
        dataSource.connection.use { connection ->
            connection.prepareStatement("SELECT label, features FROM $TABLE_NAME ORDER BY id").use { statement ->
                statement.executeQuery().use { rows ->
                    while (rows.next()) {
                        val features = codec.decode(encryption.decrypt(rows.getBytes("features")))
                        observations.add(LabeledFeatures(label = Label(rows.getString("label")), features = features))
                    }
                }
            }
        }
        return observations
    }

    override fun labels(): Set<Label> {
        val labels = LinkedHashSet<Label>()
        dataSource.connection.use { connection ->
            connection.prepareStatement("SELECT DISTINCT label FROM $TABLE_NAME").use { statement ->
                statement.executeQuery().use { rows ->
                    while (rows.next()) {
                        labels.add(Label(rows.getString("label")))
                    }
                }
            }
        }
        return labels
    }

    override fun size(): Int {
        dataSource.connection.use { connection ->
            connection.prepareStatement("SELECT COUNT(*) FROM $TABLE_NAME").use { statement ->
                statement.executeQuery().use { rows ->
                    rows.next()
                    return rows.getInt(1)
                }
            }
        }
    }

    override fun clear() {
        dataSource.connection.use { connection ->
            connection.prepareStatement("DELETE FROM $TABLE_NAME").use { statement ->
                statement.executeUpdate()
            }
        }
    }

    private companion object {
        const val TABLE_NAME = "skein_feature_observation"
        const val INSERT_SQL = "INSERT INTO $TABLE_NAME (label, features) VALUES (?, ?)"
    }
}
