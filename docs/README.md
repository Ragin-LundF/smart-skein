# Skein documentation

Each module has a short `README.md` covering what it is and when to reach for it. This directory
holds the depth: how each part works, how to use it, and how to integrate it.

## Start here

| | |
|---|---|
| [Getting started](getting-started.md) | Install, train a model, score a record. About ten minutes. |
| [Architecture](architecture.md) | Modules, layers, and how data moves through them. |
| [Bring your own data](bring-your-own-data.md) | Six steps from your records to a model you can defend. |
| [Migrating to 2.0.0](migration.md) | What a 1.2.0 consumer has to change. Short: recompile. |

## By module

| Module | Documentation |
|---|---|
| [`skein-text`](../skein-text) | [Overview](text/README.md) · [Normalization](text/normalization.md) · [Tokenization](text/tokenization.md) · [Signatures](text/signatures.md) · [Word repair](text/word-repair.md) |
| [`skein-classify`](../skein-classify) | [Overview](classify/README.md) · [Schema](classify/schema.md) · [Featurisation](classify/featurisation.md) · [Single-label](classify/single-label.md) · [Multi-label](classify/multi-label.md) · [Evaluation](classify/evaluation.md) · [Persistence](classify/persistence.md) · [Scale](classify/scale.md) |
| [`skein-classify-embedding-onnx`](../skein-classify-embedding-onnx) | [Overview](embeddings/README.md) · [Local ONNX](embeddings/onnx-local.md) · [Choosing a model](embeddings/choosing-a-model.md) |
| [`skein-classify-embedding-http`](../skein-classify-embedding-http) | [Overview](embeddings/README.md) · [External service](embeddings/external-service.md) · [Choosing a model](embeddings/choosing-a-model.md) |
| [`skein-extract`](../skein-extract) | [Overview](extract/README.md) · [Slot filling](extract/slots.md) · [Patterns](extract/patterns.md) · [Clustering](extract/clustering.md) · [CRF tagging](extract/crf.md) · [Persistence](extract/persistence.md) |
| [`skein-store-postgres`](../skein-store-postgres) | [Overview](store-postgres/README.md) · [Components](store-postgres/components.md) · [Testing](store-postgres/testing.md) |
| [`skein-cli`](../skein-cli) | [Overview](cli/README.md) · [Flags](cli/flags.md) · [Install](cli/install.md) · [Evaluate](cli/evaluate.md) · [Predict](cli/predict.md) |

## Decisions

Architecture decision records live in [`adr/`](adr). They record why something is the way it is,
which is the part that does not survive in code. Like every other page here they describe the
current state; the alternatives they name are standing reasons, not a history of the discussion.

- [ADR 0001 — Multi-label classification in `skein-classify`](adr/0001-multi-label-classification.md)
- [ADR 0002 — Verifying the identity of an external embedding service](adr/0002-vector-canary.md)
