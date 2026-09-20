# skein-text

The shared text foundation. Normalization, broken-word repair, tokenization into typed tokens, and
structural pattern signatures.

Depends on nothing. Everything else in Skein builds on it, so the bar for adding anything here is
high.

## Use it when

- Text arrives messy and you need it consistent before anything else looks at it.
- Words are broken by OCR, line wrapping or fixed-width fields: `"apart ment"` → `"apartment"`.
- You need to know the *shape* of a token — word, number, date, amount, identifier — without
  retaining its content.
- You want a structural fingerprint of a line (`<word> <date> <amount>`) to group similar records.

## At a glance

```kotlin
import io.skein.text.application.TypedTokenizer
import io.skein.text.infrastructure.DefaultTextNormalizer

val normalized = DefaultTextNormalizer().normalize(raw = "Invoice  31.12.2024   1.234,56 EUR")

TypedTokenizer().tokenize(text = normalized).forEach { token ->
    println("${token.text} : ${token.type}")   // WORD, DATE, AMOUNT, NUMERIC, ALPHANUMERIC, …
}
```

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-text")
}
```

## Documentation

**[Full documentation →](../docs/text/README.md)** — the mental model, installation and the
end-to-end flow, with a page each for
[normalization](../docs/text/normalization.md),
[typed tokenization](../docs/text/tokenization.md),
[pattern signatures](../docs/text/signatures.md) and
[broken-word repair](../docs/text/word-repair.md).

Related: [Architecture](../docs/architecture.md) · [All documentation](../docs/README.md)
