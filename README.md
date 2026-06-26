# Skein

> A self-learning, privacy-preserving toolkit for recognizing, classifying and extracting
> structure from messy text and records on the JVM. No GPU, no external AI, no neural
> networks — pure classical statistical machine learning, fully on the CPU.

The name (a coiled strand of yarn) reflects the purpose: *untangling a skein* of unstructured
text into recognizable patterns. Each module is one thread that builds on the others.

## Building blocks

1. **Shared text foundation** — normalization, broken-word repair (`"apart ment" → "apartment"`),
   and a typed tokenizer producing pattern signatures (`<word> <date> <numeric>`).
2. **Classification** — assign a label to a whole record, typo-tolerant, self-training, with
   privacy-preserving feature hashing so personal data is never stored in clear text.
3. **Extraction** — pull structured values out of text via typed-token patterns and slot filling.

## Module layout

```
                 skein-text  (pure foundation, zero heavy deps)
                  ╱        ╲
        skein-classify    skein-extract
              │
     skein-store-postgres
```

| Module | Published | Responsibility |
|--------|-----------|----------------|
| [`skein-bom`](skein-bom) | yes | Version alignment for consumers (Bill of Materials) |
| [`skein-text`](skein-text) | yes | Shared text foundation: normalization, broken-word repair, typed tokenizer, pattern signatures |
| [`skein-classify`](skein-classify) | yes | Record → label classification (Naive Bayes + logistic regression, active learning) |
| [`skein-extract`](skein-extract) | yes | Text → structured field extraction (pattern DSL, slot filling, CRF token tagging) |
| [`skein-store-postgres`](skein-store-postgres) | yes | Optional PostgreSQL persistence adapter (AES-256-GCM at rest) |
| [`examples`](examples) | no | Runnable samples, including transaction categorization |

## End-to-end example

`./gradlew :examples:run` demonstrates **classify → route → extract** over bank transactions —
see [`examples`](examples).

## Publishing

Published modules ship a main jar, a sources jar and a Dokka-generated javadoc jar with full POM
metadata (MIT license, SCM), to Maven Central (Sonatype Central Portal, GPG-signed via JReleaser)
and GitHub Packages. `skein-bom` aligns all module versions for consumers.

## Toolchain

- Kotlin 2.3 (K2), JDK 25, Gradle 9.6 (wrapper committed).
- Versions live in [`gradle/libs.versions.toml`](gradle/libs.versions.toml).
- Shared build config under [`config/gradle`](config/gradle).

## License

[MIT](LICENSE).

---
