# Skein documentation

Each module has a short `README.md` covering what it is and when to reach for it. This directory
holds the depth: how each part works, how to use it, and how to integrate it.

## Start here

| | |
|---|---|
| [Getting started](getting-started.md) | Install, train a model, score a record. About ten minutes. |
| [Architecture](architecture.md) | Modules, layers, and how data moves through them. |
| [Bring your own data](bring-your-own-data.md) | Six steps from your records to a model you can defend. |

## By module

| Module | Documentation |
|---|---|
| [`skein-text`](../skein-text) | [Text foundation](text/README.md) |
| [`skein-classify`](../skein-classify) | [Overview](classify/README.md) · [Schema](classify/schema.md) · [Featurisation](classify/featurisation.md) · [Single-label](classify/single-label.md) · [Multi-label](classify/multi-label.md) · [Evaluation](classify/evaluation.md) · [Persistence](classify/persistence.md) · [Scale](classify/scale.md) |
| [`skein-classify-embedding-onnx`](../skein-classify-embedding-onnx) | [Overview](embeddings/README.md) · [Local ONNX](embeddings/onnx-local.md) · [External service](embeddings/external-service.md) · [Choosing a model](embeddings/choosing-a-model.md) |
| [`skein-extract`](../skein-extract) | [Extraction](extract/README.md) |
| [`skein-store-postgres`](../skein-store-postgres) | [PostgreSQL storage](store-postgres/README.md) |
| [`skein-cli`](../skein-cli) | [Command line](cli/README.md) |

## Decisions

Architecture decision records live in [`adr/`](adr). They record why something is the way it is,
which is the part that does not survive in code.

- [ADR 0001 — Multi-label classification in `skein-classify`](adr/0001-multi-label-classification.md)
