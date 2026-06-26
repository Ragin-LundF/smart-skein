# Static Analysis and Formatting

Use this file for Detekt, SonarQube, ktlint, `.editorconfig`, and lint cleanup.

## Tools

- Use Detekt for Kotlin static analysis.
- Use SonarQube standard rules where configured by the host project.
- Use the repository `.editorconfig` for formatting.
- The bundled `config/detekt/detekt.yml` contains project-specific overwrites and should be merged with the host project config when adopted.

## Detekt baseline expectations

The bundled Detekt overwrites include:

- `ReturnCount` active, guard clauses excluded, max returns `5`.
- `MagicNumber` ignores annotations.
- `ForbiddenComment` disabled.
- `NamedArguments` active, allowed arguments `15`, and matching names are not ignored.
- `LongParameterList` ignores default parameters, allows `25` function parameters and `15` constructor parameters.
- `TooManyFunctions` allows `20` functions per class/interface/object.
- `SpreadOperator` disabled because Kotlin compiler optimizations reduce the concern.

Do not treat these relaxed thresholds as permission to write unfocused code. They exist to reduce noise in real projects.

## SonarQube expectations

- Do not introduce code smells, duplicated logic, security hotspots, or reliability issues.
- Refactor duplicate code into meaningful abstractions only when the abstraction improves clarity.
- Avoid suppressing Sonar findings. If suppression is unavoidable, keep it local and document why.

## Suppression policy

Suppression is a last resort.

Before suppressing:

1. Understand the finding.
2. Try a small design or readability improvement.
3. Confirm behavior remains tested.
4. Add the narrowest possible suppression.
5. Document why the suppression is correct.

Never suppress findings to hide generated bad code, missing tests, security weaknesses, or rushed implementation.

## Cleanup policy

- Preserve behavior before refactoring style.
- Separate mechanical formatting changes from behavioral changes where possible.
- Do not reformat unrelated files unless the task is explicitly a formatting-only cleanup.
- Run or request the relevant Gradle/Maven checks before declaring completion.

## Expected verification commands

Use the commands that exist in the host repository. Typical options:

```bash
./gradlew test detekt
./gradlew check
mvn test
mvn verify
```

If commands cannot be run, state exactly which commands were not run and why.
