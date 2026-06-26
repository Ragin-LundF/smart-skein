package io.skein.store.postgres.config

/** Connection settings for the pooled PostgreSQL [javax.sql.DataSource]. */
data class JdbcConnectionConfig(
    val jdbcUrl: String,
    val username: String,
    val password: String,
    val maximumPoolSize: Int = DEFAULT_MAX_POOL_SIZE,
) {

    companion object {
        const val DEFAULT_MAX_POOL_SIZE = 4
    }
}
