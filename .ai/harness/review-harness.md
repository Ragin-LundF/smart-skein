# Review Harness

Use this harness for code review, self-review, or PR preparation.

## Review order

1. Contract: API, events, database, configuration, public functions.
2. Architecture: layer ownership and dependency direction.
3. Behavior: correctness of domain and application logic.
4. Security: authentication, authorization, data leakage, logging.
5. Tests: coverage, meaningful assertions, no cheating.
6. Static analysis: Detekt, SonarQube, formatting.
7. Maintainability: names, function size, duplication, comments.

## Findings format

For each finding, provide:

- Severity: blocker, major, minor, nit.
- Location.
- Issue.
- Why it matters.
- Concrete fix.

## Approval criteria

Approve only when:

- Behavior matches the requested change.
- Relevant tests exist and are meaningful.
- REST changes have Cucumber coverage.
- Static analysis and formatting are addressed.
- No security or compatibility regression is visible.
