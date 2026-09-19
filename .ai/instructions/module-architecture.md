# Module and Layer Architecture

Read this for changes that touch module boundaries, public APIs, package layout, adapters, or the
Gradle module structure.

This file states **responsibilities and rules**, not an inventory of types. Class and function names
change; the boundaries below should not.

## Layers inside a module

Every module uses the same four packages, and dependencies point inward only.

| Package | Holds | May depend on |
|---|---|---|
| `domain` | Values, entities, invariants and pure computation | Nothing outside its own `domain` |
| `spi` | Ports the module needs someone else to satisfy | `domain` only |
| `application` | Services that orchestrate domain and ports into a use case | `domain`, `spi` |
| `infrastructure` | Adapters: persistence, file formats, algorithms, external systems | `domain`, `spi`, `application` |

Rules:

- `domain` must not import `spi`, `application` or `infrastructure`. This is enforced by an
  architecture test that scans every module's production sources, so a violation fails the build.
- `spi` must not import `application` or `infrastructure`.
- Prefer adding an `infrastructure` adapter behind an existing port over widening an `application`
  service.
- Put pure computation in `domain`, even when it is only used by one service. Logic that needs no
  collaborator is far cheaper to test there, and coverage gates make that difference visible.
- Put configured, stateful collaborators in `application`.
- A type that describes *how a specific implementation works* belongs in `infrastructure`, not
  `domain`, even when it is a plain value. Putting it in `domain` teaches the innermost layer about
  an outer one — a real regression that an import-direction check will not catch.
- Presentation helpers that only transform domain values into text are `application`, not
  `infrastructure`: they perform no I/O.

## Rules across modules

- A module declares its public surface with `api(...)` only when a consumer needs those types in its
  own signatures; otherwise use `implementation(...)`.
- Do not add a dependency to a module that currently has none without a specific reason. A new
  dependency on a published module becomes a transitive dependency for every downstream consumer.
  Prefer the standard library first.
- Ports live in the module that *needs* them, not the module that implements them. An adapter module
  depends on the module owning the port, never the reverse.
- Keep persistence formats owned by the module whose data they describe.
- A published module's public API is a contract. Prefer additive change. When adding to an
  interface, give the new member a default implementation so existing implementors keep compiling
  and linking. This relies on the build compiling Kotlin interface defaults to real JVM default
  methods; if that setting is ever removed, every such addition becomes a hard break. Record every
  unavoidable break in the changelog, including on-disk format changes.
- Adding a property to a published data class breaks its constructor, `copy` and `componentN`
  signatures. Adding a *method* does not. Prefer a method, or a new overload, over changing an
  existing signature.
- Never let one module reach into another's internals to avoid an API change. Either widen the API
  deliberately or move the logic.

## Modules in this repository

```
                 skein-text  (foundation, no dependencies)
                  ╱        ╲
        skein-classify    skein-extract
              │
     skein-store-postgres
```

| Module | Published | Responsibility |
|---|---|---|
| `skein-text` | yes | The shared text foundation: normalization, broken-word repair, tokenization into typed tokens, and structural pattern signatures. Depends on nothing. Everything else builds on it. |
| `skein-classify` | yes | Assigning labels to a whole record — one label, or several when labels co-occur: schema definition and validation, privacy-preserving feature hashing, the learning algorithms, model persistence, active-learning support, calibration, explanation and quality evaluation. Single-label scoring is the `Classifier` port; co-occurring labels are `MultiLabelClassifier` with `BatchLearner`, and the two are siblings rather than alternatives. See `docs/adr/0001-multi-label-classification.md`. |
| `skein-extract` | yes | Pulling structured values *out* of text: typed-token patterns, slot filling, layout clustering, and a trainable token tagger with its own model format. Unlike classification, it returns real values rather than hashes. |
| `skein-store-postgres` | yes | An optional storage adapter implementing the classification module's storage port against PostgreSQL, with encryption at rest. Contains no learning logic. |
| `skein-cli` | yes | Command-line tools over the library: interactive labeling, batch prediction, model inspection and evaluation. Holds only CLI concerns — argument parsing, CSV I/O, terminal interaction — and no algorithm that belongs in a library module. |
| `skein-bom` | yes | Version alignment for consumers. Contains no code. |
| `examples` | no | Runnable demonstrations, one per capability. Not published, and nothing else may depend on it. |

Guidance:

- New text handling shared by both classification and extraction goes in `skein-text`. Anything
  placed there is a dependency of the whole library, so the bar is high.
- Do not let `skein-classify` and `skein-extract` depend on each other. They are siblings that meet
  only in a consumer such as `examples`.
- Logic needed by more than one consumer belongs in a library module, even when it looks like
  presentation. Duplicating it across consumers that cannot share code is the failure mode to avoid.
- The CLI is a consumer, not a place to grow features. If a CLI command needs new behaviour, add it
  to the library module that owns the concept and have the command call it.

## Privacy boundaries

The library's privacy properties are not uniform, and any change that moves data across a module
boundary must respect that.

- Classification features are irreversible keyed hashes. Nothing in that path may introduce a stored
  mapping back to source text, because that would turn a model file into a lookup table for the
  training corpus.
- Extraction deliberately returns real values, and a trained tagger's learned features can contain
  fragments of the training text. Anything persisting that data must expose the trade-off explicitly
  and default to the safer option.
- Fields marked as sensitive must not reach feature extraction.
- When a change weakens a documented privacy property, update the documentation in the same change.
  Never leave a claim in place that the code no longer supports.

## Adding a module

Before adding one, check whether the responsibility belongs in an existing module. A new module is
justified when it introduces an optional dependency that library users should be able to avoid, or
an independently versioned deliverable.

A new module must: use the four-package layout, declare its own coverage and static-analysis gates
through the shared convention plugins, carry a README describing its responsibility, and be added to
the BOM if it is published.
