# skein-store-postgres

Optional **PostgreSQL persistence** for Skein: implements `skein-classify`'s `FeatureStore` SPI on
PostgreSQL, with optional **AES-256-GCM encryption at rest**. Without this module a
`ClassificationService` keeps its corpus only in memory (lost on restart); with it, every learned
observation is durable and survives across runs — which is what makes `retrain` and `metrics`
meaningful in production. Depends on `skein-classify`.

> **Audience:** developers deploying Skein with a real database, and anyone who needs the learned
> corpus to outlive the process or to be encrypted at rest.

## Pages

| | |
|---|---|
| [Components](components.md) | The store, the codec, encryption at rest, connection handling, schema migration |
| [Testing](testing.md) | Testcontainers, what needs Docker and what does not |

## When you need it

`ClassificationService` persists every observation (label + feature vector) to a `FeatureStore`.
The default `InMemoryFeatureStore` is fine for tests and short-lived jobs. Use `PostgresFeatureStore`
when you need:

- **Durability** — the training corpus survives restarts; `retrain(epochs)` replays it from disk.
- **Encryption at rest** — for `PrivacyModeEnum.ENCRYPTED_SOURCE`, source-derived data is stored as
  ciphertext, not plaintext.
- **Shared corpus** — multiple processes pointing at one database.

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-store-postgres")
    // brings in the PostgreSQL driver, Tomcat JDBC pool, and Liquibase transitively
}
```

## Quick start

```kotlin
// 1. Build a pooled DataSource.
val dataSource = TomcatJdbcDataSourceFactory.create(
    JdbcConnectionConfig(
        jdbcUrl  = "jdbc:postgresql://localhost:5432/skein",
        username = "skein",
        password = secret,
        // maximumPoolSize = 4   (default)
    ),
)

// 2. Create the store. The table is created/migrated idempotently on construction.
val store = PostgresFeatureStore(dataSource)                       // plaintext (features only)
//          PostgresFeatureStore(dataSource, encryption = AesGcmEncryption(key32Bytes))  // encrypted

// 3. Plug it into the classification engine via the featureStore param.
val engine = ClassificationService(
    schema,
    PrivacyModeEnum.FEATURES_ONLY,        // or ENCRYPTED_SOURCE with AesGcmEncryption
    hashingConfig,
    featureStore = store,
)
```

From here, `engine.learn`, `retrain`, `metrics`, and `forget` all read/write PostgreSQL transparently.

## Package layout

```
io.skein.store.postgres
├─ config/          JdbcConnectionConfig
├─ spi/             FeatureEncryption (port)
└─ infrastructure/  PostgresFeatureStore, FeatureVectorCodec, SchemaMigrator,
                    NoEncryption, AesGcmEncryption, TomcatJdbcDataSourceFactory
```

---

[Components](components.md) · [Testing](testing.md) · [All documentation](../README.md)
