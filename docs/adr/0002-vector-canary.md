# ADR 0002 — Verifying the identity of an external embedding service

- **Status:** accepted
- **Date:** 2026-09-20
- **Affects:** `skein-classify` public API (`Vectorizer`, `ModelStore`), `.skein` payload `0x03`, `skein-classify-embedding-http`

## Context

[ADR 0001 §5](0001-multi-label-classification.md) makes a featurisation mismatch fatal on load: a
model scored with the wrong vectorizer returns confident, wrong labels with nothing in the output to
notice, so `ModelStore.loadMultiLabel` refuses it.

That check compares `VectorizerFingerprint`, which covers what a vectorizer can *declare about
itself*. The guarantee it provides is therefore only as strong as what the vectorizer knows:

| Vectorizer | What the fingerprint covers | Strength |
|---|---|---|
| `HashingVectorizer` | key, width, n-gram ranges, weighting, normalizer | Complete — the output is a pure function of these |
| `OnnxEmbeddingVectorizer` | a digest of the **model file's bytes**, plus tokenizer, pooling, normalisation | Complete — a model swapped in place changes the digest |
| `HttpEmbeddingVectorizer` | model name, declared revision, prefix, width | **Incomplete** |

An embeddings endpoint returns vectors. Nothing in the protocol says which weights produced them.
So this sequence passes every check the library has and produces wrong labels:

1. A model is trained against `multilingual-e5-small` served by LM Studio.
2. Someone updates the model in LM Studio.
3. `modelRevision` is a label a human maintains, and still says `@1`.
4. The fingerprint matches, `loadMultiLabel` accepts the model, inference returns confident wrong
   labels, and nothing logs.

`modelRevision` is a declaration. It is worth having, and it is not a verification.

## Decision

### 1. Sample what the vectorizer does, not only what it declares

`VectorizerCanary` holds fixed probe texts and the vectors a vectorizer produced for them. The
probes travel inside the model file and are re-embedded on load; if they have moved, the load
throws `VectorizerCanaryException`.

It lives in `skein-classify` beside the fingerprint it extends, not in the HTTP adapter, for two
reasons. `ModelStore.decodeMultiLabel` is the only place that holds both the stored reference and a
live `Vectorizer` in the same instant — anywhere else is a check a caller can forget, and "fails
loudly" degrades into "fails loudly if you remember". And the mechanism is vectorizer-agnostic: it
uses only `Vectorizer.vectorize`, so it works for any implementation, including third-party ones.

`Vectorizer.canary()` has a default implementation returning `null`. Implementations whose output
is a pure function of their configuration gain nothing from it and should leave it alone.

### 2. Opt in at save, verify by default at load

A vectorizer that offers no canary produces a model file that behaves exactly as it did before. A
vectorizer that offers one has already accepted the cost, so `loadMultiLabel` verifies by default
rather than requiring a second opt-in the caller can forget. `verifyCanary = false` is the escape
hatch.

The accepted cost is real and worth stating plainly: **with a canary present, opening a model file
performs network I/O** — one round trip per probe — and can fail because a service is down. Reading
a file is no longer a pure file operation. That is the trade: a loud failure at load instead of a
silent one at inference.

### 3. Relative L2 distance, not cosine

Drift is `‖observed − reference‖ / ‖reference‖`, with a default tolerance of `0.01`.

Cosine distance is the obvious choice and the wrong one. It measures direction alone, and a service
that starts L2-normalising its output — a toggle in both LM Studio and Text Embeddings Inference —
rescales every vector while rotating none. Cosine sees nothing. For a linear classifier trained on
the unnormalised vectors, that is a model-breaking change.

Relative L2 catches rotation and rescaling both, and being dimensionless it means the same thing at
384 dimensions and at 1024. For two unit vectors it reduces to `sqrt(2(1 − cos))`, so it degrades to
cosine behaviour exactly where cosine is adequate.

The tolerance sits in a gap of roughly two orders of magnitude: floating-point accumulation order
and batch composition move a vector by about `1e-5`, while a requantised or retrained model moves it
by `1e-2` or more. A model requantised from `fp16` to `q8_0` trips the check, and that is a true
positive — for a classifier trained on its vectors, a requantised model is a different model.

### 4. Capture twice: before training and again at save

`saveMultiLabel` re-checks the canary against the live vectorizer before writing, and refuses to
save if it has already moved.

A single capture proves the service returns the same vectors *now* as when the canary was taken. It
says nothing about the whole training run. Without the second check, a model swapped mid-run leaves
half the corpus embedded by one model and half by another, the canary taken beforehand still matches
at load, and the result is a model that is quietly garbage and passes every check.

### 5. The payload shape does not change

The canary occupies four appended, defaulted fields on the existing `0x03` payload. It is not a new
shape — it is optional data on the same fitted weight matrix — and `0x03` already uses this pattern
for `IdfVectorizer`'s document-frequency table. A `0x04` would make readers reject files they can
read perfectly.

## Consequences

**Gained.** Route B is usable in production. The one failure mode that the library could previously
only document is now detected, and detected at the moment a deployment can act on it.

**Cost.** Loading a model can perform network I/O and can fail for reasons unrelated to the model.
`skein-classify` carries a concept most of its vectorizers do not need. A transport failure must be
distinguished from drift, so `VectorizerCanaryException` is thrown only for drift and a timeout
propagates as itself.

**Privacy.** Probe texts are stored in the model file in clear text. They are the first thing in a
`.skein` file that is not an irreversible hash, which makes the guidance part of the design rather
than a footnote: probes must be short synthetic sentences written for the purpose, never records
from the corpus. `VectorizerCanary` enforces a length limit as a nudge, and the documentation says
so everywhere the feature appears. A `.skein` file was already documented as something to treat as
secret, so this does not change how one must be handled — but it does change what is in it.

**Route A remains the stronger guarantee.** A canary detects a changed model; a content hash makes
one impossible to miss in the first place. `skein-classify-embedding-onnx` is still the
recommendation where the choice is open.
