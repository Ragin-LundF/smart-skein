# skein-cli

Command-line tools over the library: interactive labelling, batch prediction, model inspection and
evaluation.

Holds only CLI concerns — argument parsing, CSV I/O, terminal interaction — and no algorithm that
belongs in a library module.

## Use it when

- You want to label a corpus without writing code, with the model picking what to ask about next.
- You need batch predictions over a CSV.
- You want an evaluation report from a saved model, including as a CI gate.
- You want to inspect what is inside a `.skein` file.

## At a glance

```bash
skein label    --schema schema.json --input pool.csv --model model.skein
skein predict  --model model.skein --input records.csv --output labelled.csv --min-confidence 0.8
skein evaluate --model model.skein --input holdout.csv --min-accuracy 0.9
```

`evaluate` exits with code 2 when the gate fails, so it drops straight into a pipeline.

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-cli")
}
```

## Documentation

**[Full documentation →](../docs/cli/README.md)** — every command and flag, the CSV contract, the
active-learning loop, persistence and privacy notes, and behaviour on large pools.

Related: [Evaluation](../docs/classify/evaluation.md) · [All documentation](../docs/README.md)
