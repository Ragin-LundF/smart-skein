# Low-level matching


When you need "regex over token *types*" rather than named slots, build a `TokenPattern` and find its
spans. This is the matching primitive slots are built on; use it directly to locate or validate
structure.

```kotlin
val pattern = TokenPattern.of {
    type(TokenTypeEnum.WORD)            // exactly one
    optional(TokenTypeEnum.SYMBOL)      // zero or one
    oneOrMore(TokenTypeEnum.AMOUNT)     // one or more (greedy)
    zeroOrMore(TokenTypeEnum.WORD)      // zero or more (greedy)
}

val matcher = PatternMatcher()
matcher.findAll("sum 1.00 2.00 3.00", pattern)   // [0..3]  — token-index IntRanges, leftmost & non-overlapping
matcher.matchesFully(tokens, pattern)            // true only if the pattern spans ALL tokens
```

### DSL quantifiers — `QuantifierEnum`

| Builder call | Quantifier | Meaning |
|--------------|-----------|---------|
| `type(t)` | `ONE` | exactly one token of type `t` |
| `optional(t)` | `OPTIONAL` | zero or one |
| `zeroOrMore(t)` | `ZERO_OR_MORE` | zero or more (greedy, backtracks) |
| `oneOrMore(t)` | `ONE_OR_MORE` | one or more (greedy, backtracks) |

Matching is **greedy with backtracking**: a `oneOrMore`/`zeroOrMore` element consumes as much as it
can, then gives tokens back if the rest of the pattern would otherwise fail. `findAll` returns
inclusive token-index ranges; non-matching input yields an empty list (never an error).

---

[Extraction](README.md) · [Slot filling](slots.md) · [Patterns](patterns.md) · [Clustering](clustering.md) · [CRF tagging](crf.md) · [Persistence](persistence.md) · [All documentation](../README.md)
