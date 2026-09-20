# skein-bom

A Bill of Materials that aligns the versions of every published Skein module. Contains no code.

## Use it

Import the platform once, then name modules without versions:

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))

    implementation("io.github.ragin-lundf:skein-classify")
    implementation("io.github.ragin-lundf:skein-text")
    implementation("io.github.ragin-lundf:skein-classify-embedding-onnx")
}
```

Maven:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>io.github.ragin-lundf</groupId>
      <artifactId>skein-bom</artifactId>
      <version>VERSION</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>
```

Mixing versions across Skein modules is not supported — they share types and evolve together, so the
BOM is the intended way to depend on more than one.

## Covers

`skein-text` · `skein-classify` · `skein-extract` · `skein-classify-embedding-onnx` ·
`skein-store-postgres` · `skein-cli`

Related: [Architecture](../docs/architecture.md) · [All documentation](../docs/README.md)
