# Persisting a trained tagger


A CRF that only lives in memory loses its training on exit. `CrfModelStore` writes one to a single
file and reads it back, hyperparameters and progress included, so training resumes rather than
restarting.

```kotlin
val labeler = CrfSequenceLabeler()
repeat(200) { training.forEach { (tokens, tags) -> labeler.learn(tokens = tokens, tags = tags) } }

CrfModelStore.save(path = Path("tagger.skeincrf"), snapshot = labeler.snapshot())

val restored = CrfSequenceLabeler.from(snapshot = CrfModelStore.load(path = Path("tagger.skeincrf")))
restored.label(tokens = tokens)                    // same output as the original
restored.learn(tokens = more, tags = moreTags)     // and training continues from the saved step
```

`CrfModelStore.metadata(path)` reads the header alone — retention policy, writer version, creation
time — without inflating the weights.

### ⚠️ A saved CRF contains your training text

This is the one place where Skein's "never stored in clear text" property does **not** hold. A CRF
learns features keyed by the token text itself (`word=`, plus three-character `prefix=`/`suffix=`
affixes), so the weight table is a lexicon of the training corpus. Anyone with the file can
decompress it and read that vocabulary. `skein-classify` is different — its features are
irreversible keyed hashes.

| Retention | What is written | Privacy | Cost |
|---|---|---|---|
| `ALL_FEATURES` | every weight | **none** — every token seen even once is on disk | the only mode that restores a bit-identical model |
| `FREQUENT_ONLY` *(default)* | lexical features seen at least `minLexicalOccurrences` times (default 2), plus all structure | **reduced, not eliminated** — a name occurring twice survives | negligible; a weight from one occurrence is statistically worthless anyway |
| `STRUCTURAL_ONLY` | token-type structure and transitions only | **zero clear text** | loses generalization from word shape, e.g. recognizing an unseen `-ing` word by its suffix |

The default is the safer mode deliberately: `ALL_FEATURES` as a default would turn a
privacy-preserving library into a PII exporter the first time anyone called `save`.

### File format `SKCR`

```
offset 0   4 bytes  magic 'S' 'K' 'C' 'R'
offset 4   1 byte   format major (breaking)
offset 5   1 byte   format minor (additive)
offset 6   ...      GZIP( payload )
```

Hand-rolled over `DataOutputStream`, so this module keeps its single dependency on `skein-text`.
The reader accepts any minor version — including a newer one — and never asserts end-of-stream, so
fields a later writer appends are ignored rather than fatal. A differing major is refused with a
message naming both versions. Enum values are written as stable codes, never ordinals, so
reordering an enum constant cannot silently reinterpret an old file.

### Error behavior

- `save` throws `IllegalArgumentException` for an untrained labeler or a `minLexicalOccurrences`
  below 1.
- `load` throws `IllegalArgumentException` for a foreign magic, an unreadable major version, or a
  truncated or corrupt payload.
- `CrfSequenceLabeler.from` validates the snapshot — non-empty and duplicate-free tag order, a
  non-negative step, and no weight referencing an unknown tag.

---

[Extraction](README.md) · [Slot filling](slots.md) · [Patterns](patterns.md) · [Clustering](clustering.md) · [CRF tagging](crf.md) · [Persistence](persistence.md) · [All documentation](../README.md)
