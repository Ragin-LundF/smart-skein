# skein-store-postgres

An optional storage adapter implementing `skein-classify`'s `FeatureStore` port against PostgreSQL,
with AES-256-GCM encryption at rest.

Contains no learning logic. Depends on `skein-classify` because that is where the port lives.

## Use it when

- Training data must outlive the process. The default `InMemoryFeatureStore` is lost on restart.
- Several processes share one corpus.
- You use `PrivacyModeEnum.ENCRYPTED_SOURCE` and need the original records retained, encrypted.
- The corpus is larger than you want to hold in heap.

## At a glance

```kotlin
import io.skein.classify.application.ClassificationService
import io.skein.classify.domain.PrivacyModeEnum

val engine = ClassificationService(
    schema = schema,
    privacyMode = PrivacyModeEnum.ENCRYPTED_SOURCE,
    hashingConfig = hashingConfig,
    featureStore = postgresFeatureStore,
)
```

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-store-postgres")
}
```

Brings a connection pool and Liquibase; the PostgreSQL driver is `runtimeOnly`.

## Documentation

**[Full documentation →](../docs/store-postgres/README.md)** — schema migrations, connection
setup, the encryption key contract, and what `ENCRYPTED_SOURCE` does and does not protect.

Related: [Single-label storage](../docs/classify/single-label.md#storage) ·
[All documentation](../docs/README.md)
