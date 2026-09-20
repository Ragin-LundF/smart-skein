# Typed tokenization


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

[Text foundation](README.md) · [Normalization](normalization.md) · [Tokenization](tokenization.md) · [Signatures](signatures.md) · [Word repair](word-repair.md) · [All documentation](../README.md)
