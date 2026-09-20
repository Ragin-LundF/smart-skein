# Components


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

[PostgreSQL storage](README.md) · [Components](components.md) · [Testing](testing.md) · [All documentation](../README.md)
