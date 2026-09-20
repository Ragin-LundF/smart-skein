# Enable it locally without runtime pollution


Most consumers depend on `skein-classify` for the library and only need the CLI as a *tool*. Don't put
it on your `implementation`/`runtimeClasspath` — declare it in an **isolated configuration** and run it
via a `JavaExec` task. It stays out of your application jar entirely:

```kotlin
val skeinCli by configurations.creating

dependencies {
    skeinCli("io.github.ragin-lundf:skein-cli:<version>")
}

tasks.register<JavaExec>("skeinLabel") {
    classpath = configurations["skeinCli"]   // isolated — never in your runtime classpath
    mainClass = "io.skein.cli.MainKt"
    args(
        "label",
        "--input", "data.csv",
        "--label-col", "category",
        "--out", "labeled.csv",
        "--model", "model.skein",
    )
}
```

Then `./gradlew skeinLabel`. Nothing from `skein-cli` leaks into your shipped artifact.

---

[Command line](README.md) · [Flags](flags.md) · [Install](install.md) · [Evaluate](evaluate.md) · [Predict](predict.md) · [All documentation](../README.md)
