package io.skein.store.postgres.infrastructure

import liquibase.Contexts
import liquibase.LabelExpression
import liquibase.Liquibase
import liquibase.database.DatabaseFactory
import liquibase.database.jvm.JdbcConnection
import liquibase.resource.ClassLoaderResourceAccessor
import javax.sql.DataSource

/**
 * Applies the Liquibase changelog under `db/changelog` to the data source. Idempotent and safe to
 * run on every startup — Liquibase applies each changeSet exactly once and records it in its
 * tracking tables.
 *
 * Liquibase (rather than hand-rolled DDL) so the schema can evolve across versions with an audit trail.
 */
class SchemaMigrator(private val dataSource: DataSource) {

    fun migrate() {
        dataSource.connection.use { connection ->
            val database = DatabaseFactory.getInstance()
                .findCorrectDatabaseImplementation(JdbcConnection(connection))
            Liquibase(CHANGELOG, ClassLoaderResourceAccessor(), database).use { liquibase ->
                liquibase.update(Contexts(), LabelExpression())
            }
        }
    }

    private companion object {
        const val CHANGELOG = "db/changelog/db.changelog-master.xml"
    }
}
