# skein-text

The shared text foundation for Skein. **Pure, zero heavy dependencies, no training data shipped.**
Every other module (`skein-classify`, `skein-extract`) builds on the four primitives here:
normalize → tokenize → fingerprint → repair.

> **Audience:** developers wiring text through the pipeline, and data scientists who need to know
> exactly how raw strings become typed tokens and privacy-preserving signatures before any model
> sees them.

## Mental model

```
raw string ──normalize──▶ clean string ──tokenize──▶ List<Token> ──┬─▶ PatternSignature  (structure only)
                                                                    └─▶ feature text / extraction
                              ▲
                  BrokenWordRepairer + FrequencyModel
                  (optional repair of split words before tokenizing)
```

- **Normalization** strips noise so the same text always looks the same.
- **Tokenization** assigns each token a *structural type* (`WORD`, `DATE`, `AMOUNT`, …) — this is the
  privacy-friendly view: it captures *shape*, not content.
- **Pattern signatures** are the ordered list of those types — a content-free fingerprint you can
  safely log, compare, and cluster.
- **Broken-word repair** is a *self-learned* clean-up step (the only learnable component here): it
  re-joins words that OCR or bad exports split apart.

## Installation

```kotlin
dependencies {
    implementation("io.skein:skein-text:<version>")   // align versions via skein-bom
}
```

---

## 1. Normalization — `TextNormalizer` / `DefaultTextNormalizer`

`DefaultTextNormalizer` applies four conservative, **idempotent** steps in this exact order:

| Step | What | Example |
|------|------|---------|
| 1 | Replace control chars (`\p{Cntrl}`) with a space | `"a\tb"` → `"a b"` |
| 2 | Collapse runs of whitespace (`\s+`) to one space | `"a   b"` → `"a b"` |
| 3 | Trim leading/trailing whitespace | `" a b "` → `"a b"` |
| 4 | Lowercase | `"Apart Ment"` → `"apart ment"` |

```kotlin
val normalizer = DefaultTextNormalizer()
normalizer.normalize("  Payment\t 1234.56  ON  2024-12-31 ")   // "payment 1234.56 on 2024-12-31"

// Idempotent: normalize(normalize(x)) == normalize(x)
```

`TextNormalizer` is an SPI (port). Supply your own implementation if you need locale-specific
folding (e.g. accent stripping); the constructors that accept a normalizer
(`HashingVectorizer`, etc.) take the interface, not the concrete class.

> **Note:** normalization is deliberately conservative — it never deletes word characters or
> punctuation. Anything more aggressive (stemming, accent folding) is your call to add via a custom
> `TextNormalizer`.

---

## 2. Typed tokenization — `TypedTokenizer`

Splits text into `Token`s, each tagged with a `TokenTypeEnum`. The classifier is heuristic and
defaults to **European (notably German) financial text**, but the locale-specific patterns are
injectable — see [Configuring locale patterns](#configuring-locale-patterns--tokenpatternconfig).

```kotlin
val tokenizer = TypedTokenizer()                      // default mode: WHITESPACE
val tokens = tokenizer.tokenize("apartment 31.12.2024 AB12 67,89")
// Token(text="apartment",  type=WORD,         startOffset=0,  endOffset=9)
// Token(text="31.12.2024", type=DATE,         startOffset=10, endOffset=20)
// Token(text="AB12",       type=ALPHANUMERIC, startOffset=21, endOffset=25)
// Token(text="67,89",      type=AMOUNT,       startOffset=26, endOffset=31)
```

Each `Token` carries `text`, `type`, and the half-open source span `[startOffset, endOffset)` so you
can map a token back to the original string.

### Token types

Classification tries the **most specific pattern first** (a date also looks numeric, so date wins):

| `TokenTypeEnum` | Matches | Examples | Underlying regex |
|-----------------|---------|----------|------------------|
| `DATE` | EU / ISO calendar dates | `31.12.2024`, `31/12/24`, `2024-12-31` | `\d{1,2}[./-]\d{1,2}[./-]\d{2,4}` \| `\d{4}-\d{2}-\d{2}` |
| `AMOUNT` | money, comma-decimal | `1.234,56`, `12,50`, `12.50` | `\d{1,3}(\.\d{3})*,\d{2}` \| `\d+,\d{2}` \| `\d+\.\d{2}` |
| `NUMERIC` | integers / grouped numbers | `1234`, `1.234` | `\d+([.,]\d{3})*` |
| `WORD` | pure letters | `apartment`, `München` | `\p{L}+` |
| `ALPHANUMERIC` | letters + digits, no symbols | `AB12`, `R2D2` | `[\p{L}\d]+` |
| `SYMBOL` | only punctuation | `:`, `-->` | `[^\p{L}\d\s]+` |
| `WORD_SYMBOL` | word-like with internal symbols (fallback) | `AIG-Life`, `CustomerNumber:` | everything else |

### Tokenization modes — `TokenizationModeEnum`

The `mode` constructor knob decides how punctuation is handled:

```kotlin
TypedTokenizer(mode = TokenizationModeEnum.WHITESPACE)         // default
TypedTokenizer(mode = TokenizationModeEnum.PUNCTUATION_AWARE)
```

| Mode | Behavior | `"CustomerNumber: AB12"` becomes |
|------|----------|----------------------------------|
| `WHITESPACE` (default) | Splits **only** on whitespace; punctuation stays attached. Keeps insurer codes (`R+V`) and `key:value` pairs intact. | `[WORD_SYMBOL "CustomerNumber:"]`, `[ALPHANUMERIC "AB12"]` |
| `PUNCTUATION_AWARE` | Splits leading/trailing punctuation into separate `SYMBOL` tokens, while keeping dates/amounts/numbers whole. | `[WORD "CustomerNumber"]`, `[SYMBOL ":"]`, `[ALPHANUMERIC "AB12"]` |

**Pick `PUNCTUATION_AWARE`** when you want to anchor on a keyword and the keyword is glued to a colon;
**pick `WHITESPACE`** (default) when symbol-joined tokens like `AIG-Life` are meaningful and should
stay together.

### Configuring locale patterns — `TokenPatternConfig`

The `DATE`/`AMOUNT`/`NUMERIC` regexes are the only locale-specific part of classification (the
`WORD`/`ALPHANUMERIC`/`SYMBOL` rules are Unicode-general). They live in a `TokenPatternConfig` you
pass to the constructor, so you can swap conventions without forking the tokenizer:

```kotlin
TypedTokenizer()                                    // default = TokenPatternConfig.GERMAN
TypedTokenizer(patterns = TokenPatternConfig.US)    // mm/dd/yyyy dates, dot-decimal amounts
```

```kotlin
TypedTokenizer(patterns = TokenPatternConfig.US).tokenize("12/31/2024 1,234.56")
// Token(text="12/31/2024", type=DATE)
// Token(text="1,234.56",   type=AMOUNT)
```

`TokenPatternConfig` is just an **ordered list of `Regex → TokenTypeEnum` rules**. Order is priority:
the first full match wins for classification, and the longest anchored match wins for boundary
detection in `PUNCTUATION_AWARE` mode. Keep `DATE` and `AMOUNT` ahead of `NUMERIC` (both also look
numeric).

Build your own locale, or **add domain recognizers** that should stay one token — e.g. an order code
that punctuation-aware mode would otherwise split at the `-`:

```kotlin
val config = TokenPatternConfig(
    typedRules = listOf(Regex("[A-Z]{2}-\\d{3}") to TokenTypeEnum.ALPHANUMERIC) +
        TokenPatternConfig.GERMAN.typedRules,        // prepend so it wins, then reuse the EU rules
)
TypedTokenizer(mode = TokenizationModeEnum.PUNCTUATION_AWARE, patterns = config)
    .tokenize("AB-123")                              // → single ALPHANUMERIC token "AB-123"
```

> **Note:** rules map to the existing seven `TokenTypeEnum` values — you can recognize an IBAN or
> phone number, but the *label* must be one of the existing types (there is no custom token type).

---

## 3. Pattern signatures — `PatternSignature`

The ordered sequence of token types — a **content-free structural fingerprint**. Safe to log,
compare across records, and cluster on (`skein-extract`'s `TemplateClusterer` does exactly this).

```kotlin
val tokens = tokenizer.tokenize("payment 1234.56 on 2024-12-31")
val signature = PatternSignature.of(tokens)
signature.render()        // "<word> <amount> <word> <date>"
signature.types           // [WORD, AMOUNT, WORD, DATE]

// Two records with the same layout share a signature even if every value differs:
PatternSignature.of(tokenizer.tokenize("invoice 99,00 on 2025-01-01")).render()
// "<word> <amount> <word> <date>"  ← same fingerprint
```

It's a `@JvmInline value class` over `List<TokenTypeEnum>`, so equality/hashing work out of the box —
you can use signatures as map keys with no boxing cost.

---

## 4. Broken-word repair — `BrokenWordRepairer` + `FrequencyModel`

The only **learnable** component in `skein-text`. It re-joins words that were wrongly split
(`"apart ment"` → `"apartment"`) by choosing the word boundaries that best explain the fragments,
using a self-learned vocabulary. It **never rewrites content** — it only changes where word
boundaries fall.

### Step 1 — train a `FrequencyModel`

The model counts how often each word is seen (case-insensitive). It has a **privacy threshold**,
`minKeepFrequency`: words seen fewer than that many times are treated as unknown and never become
vocabulary — so rare tokens (likely personal data) cannot leak into the dictionary or its
serialization.

```kotlin
val model = FrequencyModel(minKeepFrequency = 1)      // default = 1 (keep everything seen ≥1×)
model.learnAll(listOf("apartment", "apartment", "payment", "rent", "rent", "rent"))

model.frequency("rent")        // 3
model.isKnown("apartment")     // true
model.knownWords()             // {"apartment", "payment", "rent"}
```

> **Privacy tuning (data scientists):** keep `minKeepFrequency = 1` only for dev/tests. In
> production raise it (e.g. **3–5**) so one-off names, account numbers, and other PII never cross the
> threshold into vocabulary. `frequency()` returns `0` and `isKnown()` returns `false` for anything
> below the threshold, and such words are dropped from `serialize()`.

Persist and reload a trained model:

```kotlin
val text = model.serialize()                          // threshold line + "count\tword" lines
val restored = FrequencyModel.deserialize(text)       // exact counts + threshold restored
```

### Step 2 — repair text

```kotlin
val repairer = BrokenWordRepairer(
    frequencyModel = model,
    maxFragmentsPerWord = 4,    // default: merge up to 4 consecutive fragments
    maxEditDistance = 1,        // default: tolerate 1 typo against known words
)

repairer.repair("the apart ment is ready")   // "the apartment is ready"
repairer.repair("under stand ing")           // "understanding"
repairer.repair("xyz")                        // "xyz"  (unknown single token left alone)
```

### How it decides (the algorithm)

`repair` runs **dynamic-programming re-segmentation** over the whitespace fragments, scoring every
candidate merge and keeping the segmentation with the best total score:

| Candidate | Score |
|-----------|-------|
| Known word | `10.0 + ln(frequency)` — strongly rewarded, frequent words preferred |
| Near-known (≥4 chars, within `maxEditDistance`) | `8.0` — typos tolerated, but ranked below exact matches |
| Unknown single fragment | `-1.0` — kept as-is, not forced to merge |
| Unknown merge of N fragments | `N × -1000.0` — merging into a non-word is almost always rejected |

Typo tolerance uses a **SymSpell** index (`SymSpellIndex`, precomputed deletion variants) for fast
"is this within k edits of a known word?" lookups; the index is rebuilt only when the vocabulary
grows.

### Tuning knobs (data scientists)

| Knob | Default | Raise it to… | Cost of raising |
|------|---------|--------------|-----------------|
| `maxFragmentsPerWord` | `4` | join words split into many pieces (`u n d e r`) | more DP work per text |
| `maxEditDistance` | `1` | tolerate noisier OCR/typos | more false-positive merges; slower index |
| `FrequencyModel.minKeepFrequency` | `1` | tighten privacy / shrink vocabulary | rarer real words stop being repaired |

Words shorter than 4 chars are never matched by edit distance (the `MIN_TYPO_LENGTH` guard), to avoid
spurious merges of short fragments.

---

## End-to-end usage

```kotlin
val normalizer = DefaultTextNormalizer()
val tokenizer = TypedTokenizer(mode = TokenizationModeEnum.WHITESPACE)

val model = FrequencyModel(minKeepFrequency = 3).apply {
    learnAll(corpus.flatMap { it.split(' ') })       // self-learn vocabulary from your corpus
}
val repairer = BrokenWordRepairer(model)

val clean = repairer.repair(normalizer.normalize(rawText))
val tokens = tokenizer.tokenize(clean)
val signature = PatternSignature.of(tokens).render()
```

This `clean → tokens → signature` flow is exactly what `skein-classify` (feature text) and
`skein-extract` (slot matching, clustering) consume.

---

## Package layout

```
io.skein.text
├─ domain/          Token, TokenTypeEnum, TokenizationModeEnum, TokenPatternConfig, PatternSignature, FrequencyModel
├─ spi/             TextNormalizer (port)
├─ application/     TypedTokenizer, BrokenWordRepairer, SymSpellIndex
└─ infrastructure/  DefaultTextNormalizer
```

| Type | Layer | Key API |
|------|-------|---------|
| `TextNormalizer` | spi | `normalize(String): String` |
| `DefaultTextNormalizer` | infrastructure | the 4-step pipeline above |
| `TypedTokenizer` | application | `tokenize(String): List<Token>`; ctor `mode`, `patterns` |
| `Token` | domain | `text`, `type`, `startOffset`, `endOffset` |
| `TokenPatternConfig` | domain | `typedRules`; presets `GERMAN` (default), `US` |
| `PatternSignature` | domain | `of(tokens)`, `render()`, `types` |
| `FrequencyModel` | domain | `learn`/`learnAll`/`frequency`/`isKnown`/`knownWords`/`serialize`/`deserialize` |
| `BrokenWordRepairer` | application | `repair(String): String`; ctor knobs |

> **Calibration reminder:** token classification is heuristic and locale-tuned. The `TokenPatternConfig`
> passed to `TypedTokenizer` and the privacy threshold in `FrequencyModel` are the knobs that adapt
> Skein to your data — treat them as the first thing to tune when results look off.

