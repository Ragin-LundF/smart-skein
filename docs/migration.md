# Migrating from 1.2.0 to 2.0.0

Two things force the major version: a data class gained a property, and multi-label models are
written in a payload shape a 1.x reader will not open. Neither needs a code change from you.

**The short version:** recompile. If you use multi-label models, know that 1.x cannot read the
files 2.0.0 writes.

## 1. Recompile — `HashingConfig` changed shape

`HashingConfig` gained `termWeighting`, appended with a default
([`HashingConfig.kt`](../skein-classify/src/main/kotlin/io/skein/classify/domain/HashingConfig.kt)).

```kotlin
// Both still compile, unchanged:
HashingConfig(key0 = secret0, key1 = secret1)
HashingConfig(key0 = secret0, key1 = secret1, numFeatures = 1 shl 20)
```

It is a `data class`, so the generated `copy` and `componentN` signatures moved with it. Source is
unaffected; **bytecode compiled against 1.2.0 is not**. Code that never recompiles fails at link
time with a `NoSuchMethodError`, typically on `copy`.

The same applies to `LoadedMultiLabelModel`, which gained `canary`, and to
`ModelStore.loadMultiLabel`, which gained a defaulted `verifyCanary` parameter.

Recompiling against 2.0.0 is the entire migration.

> From 2.0.0 on, this class of change cannot reach a release unnoticed: every published module
> carries an ABI dump under `<module>/api/`, and `checkKotlinAbi` fails the build when the public
> surface moves. The 1.2.0 break was found by reading a diff.

## 2. Multi-label model files are not readable by 1.x

`.skein` files carry a magic number, a one-byte version, and a GZIPped ProtoBuf payload. **The
version byte selects the payload shape, not a revision**
([`ModelStore.kt`](../skein-classify/src/main/kotlin/io/skein/classify/application/ModelStore.kt)):

| Version | Payload | Written by |
|---|---|---|
| `0x01`, `0x02` | a replayable corpus of observations | single-label models, unchanged |
| `0x03` | an already-fitted weight matrix | multi-label models, new in 2.0.0 |

- **Single-label models are unaffected.** Files you wrote with 1.x load in 2.0.0 exactly as before,
  and 2.0.0 keeps writing `0x02` for them.
- **A 1.2.0 reader rejects a v3 file** at the header rather than misreading it. That is the
  intended behaviour, but the error it gives is its own generic version mismatch, not a helpful
  one — 1.2.0 predates the format and cannot describe it.
- Multi-label models are read with `ModelStore.loadMultiLabel`, not `load`. Calling `load` on a v3
  file says so explicitly.

There is no converter, and there should not be: a v3 file holds fitted weights and a v1/v2 file
holds a corpus to replay. Neither can be turned into the other without the training data.

## 3. Featurisation mismatch is now fatal

`loadMultiLabel` takes the vectorizer you intend to score with and refuses the model if it is not
the one that trained it, throwing `VectorizerMismatchException`.

This is deliberately fatal rather than a warning. A mismatched featurisation does not fail on its
own: the indices still resolve, the weights still multiply, and the labels come back confident and
wrong for as long as nobody notices.

If this throws on an upgrade, it is telling you something true. Rebuild the vectorizer with the
settings the model was trained with — `LoadedMultiLabelModel.hashingConfig` carries them when the
model used feature hashing.

## 4. If you copied `HttpEmbeddingVectorizer` out of `examples`

It is now a published module, and the copy-it-yourself instructions are gone:

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:2.0.0"))
    implementation("io.github.ragin-lundf:skein-classify-embedding-http")
}
```

The package changed from `io.skein.examples.embedding` to
`io.skein.classify.embedding.http.{domain,infrastructure}`. Two things you may have hand-patched
into your copy are now configuration rather than edits:

| You probably patched | Now |
|---|---|
| an `Authorization` header into the request builder | `EmbeddingServiceConfig.headers` |
| the `HttpClient` for a proxy or TLS context | pass your own to `JdkHttpEmbeddingTransport`, or implement `EmbeddingTransport` |

Keeping your copy is fine — that was always the deal. It will not get the width check or the canary.

## 5. Optional: turn on the canary

Nothing requires this, and a model saved without it behaves exactly as it did.

If you use an external embedding **service**, it closes the one hole the fingerprint cannot:
a model updated on the server keeps the same name, the same declared revision and the same width,
so the fingerprint matches — and the vectors are different, and the labels are quietly wrong.

```kotlin
EmbeddingServiceConfig(
    baseUrl = "http://localhost:1234/v1",
    model = "text-embedding-multilingual-e5-small",
    modelRevision = "lmstudio/multilingual-e5-small@1",
    inputPrefix = "passage: ",
    dimension = 384,
    canaryProbes = EmbeddingProbes.DEFAULT,   // <- the whole change
)
```

Two consequences before you enable it:

- **`loadMultiLabel` now performs network I/O** — one round trip per probe — and can fail because
  the service is unreachable. Pass `verifyCanary = false` to skip it.
- **Probe texts are stored in the model file in clear text.** They are the one thing in a `.skein`
  file that is not an irreversible hash. Use short synthetic sentences, never records from your
  corpus.

Existing models do not gain a canary retroactively. Re-save to capture one.

## Not affected

- `skein-text`, `skein-extract` and `skein-store-postgres` public API.
- Single-label classification: `Classifier`, `ClassificationService`, calibration, abstention,
  explanations, active learning.
- CRF models and the `SKCR` format.
- `skein-cli` commands and flags.

---

[All documentation](README.md) · [Persistence](classify/persistence.md) ·
[Changelog](../CHANGELOG.md)
