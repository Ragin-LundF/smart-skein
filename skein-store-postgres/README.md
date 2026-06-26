# skein-store-postgres

Optional PostgreSQL persistence adapter: implements `skein-classify`'s `FeatureStore` on
PostgreSQL, with optional AES-256-GCM encryption at rest. Depends on `skein-classify`.

## What it does

- **`PostgresFeatureStore`** — implements `FeatureStore` (`add` / `all` / `labels` / `size` /
  `clear`). Feature vectors are serialized with `FeatureVectorCodec` and stored as `BYTEA`. The
  table is created idempotently on construction (`SchemaMigrator`).
- **Encryption at rest** (`FeatureEncryption`): `NoEncryption` for
  `PrivacyModeEnum.FEATURES_ONLY` (features are already irreversible) and `AesGcmEncryption`
  (AES-256-GCM, random IV per write, key injected and never stored in the DB) for
  `PrivacyModeEnum.ENCRYPTED_SOURCE`.
- **Connection pool** — `JdbcConnectionConfig` + `TomcatJdbcDataSourceFactory` (Apache Tomcat
  JDBC Pool).

## Quick start

```kotlin
val dataSource = TomcatJdbcDataSourceFactory.create(
    JdbcConnectionConfig(jdbcUrl = "jdbc:postgresql://localhost:5432/skein", username = "skein", password = secret),
)
val store = PostgresFeatureStore(dataSource, encryption = AesGcmEncryption(key32Bytes))

// Plug into the classification engine:
val engine = ClassificationService(schema, PrivacyModeEnum.ENCRYPTED_SOURCE, featureStore = store)
```

## Package layout

```
io.skein.store.postgres
├─ config/          JdbcConnectionConfig
├─ spi/             FeatureEncryption (port)
└─ infrastructure/  PostgresFeatureStore, FeatureVectorCodec, SchemaMigrator,
                    NoEncryption, AesGcmEncryption, TomcatJdbcDataSourceFactory
```

## Testing

Integration tests use **Testcontainers** (`postgres:16-alpine`) and require Docker. They cover
persistence round-trips and verify that the at-rest blob is genuinely ciphertext under
`AesGcmEncryption`. Unit tests cover the codec and the AES-GCM round-trip / tamper detection.
