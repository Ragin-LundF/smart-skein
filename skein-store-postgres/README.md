# skein-store-postgres

Optional **PostgreSQL persistence** for Skein: implements `skein-classify`'s `FeatureStore` SPI on
PostgreSQL, with optional **AES-256-GCM encryption at rest**. Without this module a
`ClassificationService` keeps its corpus only in memory (lost on restart); with it, every learned
observation is durable and survives across runs — which is what makes `retrain` and `metrics`
meaningful in production. Depends on `skein-classify`.

> **Audience:** developers deploying Skein with a real database, and anyone who needs the learned
> corpus to outlive the process or to be encrypted at rest.

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
    implementation("io.skein:skein-store-postgres:<version>")   // align via skein-bom
    // brings in the PostgreSQL driver, Tomcat JDBC pool, and Liquibase transitively
}
```

---

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

---

## Components

### `PostgresFeatureStore`

```kotlin
PostgresFeatureStore(
    dataSource: DataSource,
    codec: FeatureVectorCodec = FeatureVectorCodec(),
    encryption: FeatureEncryption = NoEncryption(),
)
```

Implements the full `FeatureStore` SPI:

| Method | Behavior |
|--------|----------|
| `add(observation)` | encode → encrypt → insert one row |
| `addAll(observations)` | batch-encodes outside the transaction, then inserts all rows in **one** transaction (rolls back on failure); empty input is a no-op |
| `all()` | fetch all rows ordered by `id`, decrypt + decode each → `List<LabeledFeatures>` |
| `labels()` | `SELECT DISTINCT label` → `Set<Label>` (insertion-ordered) |
| `size()` | `SELECT COUNT(*)` |
| `clear()` | delete all rows |

The constructor runs `SchemaMigrator` immediately, so the table exists before the first write.

### Connection pool — `JdbcConnectionConfig` + `TomcatJdbcDataSourceFactory`

```kotlin
data class JdbcConnectionConfig(
    val jdbcUrl: String,
    val username: String,
    val password: String,
    val maximumPoolSize: Int = 4,        // DEFAULT_MAX_POOL_SIZE
)
```

`TomcatJdbcDataSourceFactory.create(config)` returns an Apache Tomcat JDBC `DataSource` configured
with the PostgreSQL driver, `maxActive = maximumPoolSize`, validation-on-borrow, and a `SELECT 1`
validation query. Use any other `DataSource` if you prefer a different pool — `PostgresFeatureStore`
only needs a `javax.sql.DataSource`.

### Encryption at rest — `FeatureEncryption`

The SPI is two methods: `encrypt(ByteArray): ByteArray` / `decrypt(ByteArray): ByteArray`.

| Implementation | Use with | Behavior |
|----------------|----------|----------|
| `NoEncryption` (default) | `PrivacyModeEnum.FEATURES_ONLY` | pass-through — features are already an irreversible hash, so encrypting them adds nothing |
| `AesGcmEncryption(key)` | `PrivacyModeEnum.ENCRYPTED_SOURCE` | AES-256-GCM, fresh random 12-byte IV per write, 128-bit auth tag |

```kotlin
val key32Bytes: ByteArray = loadKeyFromKms()           // MUST be exactly 32 bytes (AES-256)
val store = PostgresFeatureStore(dataSource, encryption = AesGcmEncryption(key32Bytes))
```

`AesGcmEncryption` details:
- **Key**: exactly 32 bytes or the constructor throws. The key is copied into memory and **never
  written to the database** — supply it from your own secret manager / KMS on every startup.
- **Stored layout**: `[12-byte IV][ciphertext + 16-byte GCM tag]`. The IV is random per write, so
  encrypting the same vector twice yields different ciphertext.
- **Tamper detection**: GCM verifies the auth tag on decrypt; any modified byte throws
  `AEADBadTagException` rather than returning corrupt data.

> **Key management is yours.** Losing the key makes an `ENCRYPTED_SOURCE` corpus unrecoverable;
> leaking it defeats the encryption. Rotate by re-encrypting (`all()` under the old key →
> `clear()` → `addAll()` under the new key).

### Serialization — `FeatureVectorCodec`

Compact, allocation-light binary format stored in a `BYTEA` column:

```
[4-byte count][count × 4-byte int indices][count × 4-byte float values]   (big-endian)
```

Round-trips a sparse `FeatureVector` exactly. (Encryption, when enabled, wraps this byte array.)

### Schema migration — `SchemaMigrator`

Runs **Liquibase** against the changelog `db/changelog/db.changelog-master.xml`, creating:

```sql
CREATE TABLE skein_feature_observation (
    id       BIGINT  GENERATED ... PRIMARY KEY,   -- auto-increment
    label    TEXT    NOT NULL,
    features BYTEA   NOT NULL                       -- codec output, encrypted if enabled
);
```

Idempotent and safe to run on every startup — Liquibase tracks which change-sets are already applied.
`PostgresFeatureStore` invokes it from its constructor; you normally never call it directly.

---

## Testing

Integration tests use **Testcontainers** with `postgres:16-alpine` and therefore **require Docker**.
They cover:

- single-insert and ordered read-back round-trips (indices/values/labels preserved),
- `clear()` emptying the table,
- `addAll` batch insert,
- **encrypted round-trip while verifying the at-rest blob is genuinely ciphertext** (the stored bytes
  differ from the plaintext codec output under `AesGcmEncryption`).

Unit tests (no Docker) cover the codec round-trip and the AES-GCM round-trip + tamper detection.

```bash
./gradlew :skein-store-postgres:test          # needs Docker for the integration tests
```

---

## Package layout

```
io.skein.store.postgres
├─ config/          JdbcConnectionConfig
├─ spi/             FeatureEncryption (port)
└─ infrastructure/  PostgresFeatureStore, FeatureVectorCodec, SchemaMigrator,
                    NoEncryption, AesGcmEncryption, TomcatJdbcDataSourceFactory
```
</content>
