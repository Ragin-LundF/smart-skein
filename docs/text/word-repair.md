# Broken-word repair


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

[Text foundation](README.md) · [Normalization](normalization.md) · [Tokenization](tokenization.md) · [Signatures](signatures.md) · [Word repair](word-repair.md) · [All documentation](../README.md)
