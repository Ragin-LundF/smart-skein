package io.skein.store.postgres.infrastructure

import io.skein.store.postgres.config.JdbcConnectionConfig
import org.apache.tomcat.jdbc.pool.DataSource
import org.apache.tomcat.jdbc.pool.PoolProperties

/** Builds a pooled PostgreSQL [DataSource] (Apache Tomcat JDBC Pool) from a [JdbcConnectionConfig]. */
object TomcatJdbcDataSourceFactory {

    fun create(config: JdbcConnectionConfig): DataSource {
        val properties = PoolProperties()
        properties.url = config.jdbcUrl
        properties.username = config.username
        properties.password = config.password
        properties.driverClassName = POSTGRES_DRIVER
        properties.maxActive = config.maximumPoolSize
        properties.setTestOnBorrow(true)
        properties.validationQuery = "SELECT 1"
        return DataSource(properties)
    }

    private const val POSTGRES_DRIVER = "org.postgresql.Driver"
}
