# Testing Instructions

Use this file when adding, modifying, or reviewing tests.

## Mandatory testing principles

- High test coverage is mandatory for new and changed code.
- Tests must verify behavior, not merely execute lines.
- Every behavior change requires new or updated tests.
- Do not weaken, delete, skip, or rewrite tests just to make a change pass.
- Do not change production code solely to satisfy a brittle or artificial test unless the behavior change is required.
- Do not test implementation details when public behavior can be tested.
- Prefer small focused tests with clear names.

## Unit tests

- Use `kotlin.test` annotations and assertions when possible.
- Use named arguments for assertions, for example `assertEquals(expected = expectedValue, actual = actualValue)`.
- Unit-test functions and parameter behavior directly.
- Cover positive paths, negative paths, edge cases, boundary values, nullability where valid, and error paths.
- Validate exact exception types and important messages only when those messages are part of the contract.
- Use test fixtures/builders to reduce noise, but keep inputs visible enough to understand the scenario.

## REST API tests

- REST APIs must be tested through Cucumber/Gherkin scenarios using the BBD Cucumber Gherkin library approach when the library is available.
- REST API behavior tests must exercise the HTTP boundary, request/response bodies, status codes, headers, validation errors, security, paging, sorting, and compatibility behavior.
- Controller-only unit tests are not a replacement for REST API Cucumber scenarios.

## Coverage expectations

- New domain logic should have near-complete branch coverage.
- Critical business logic, validation, security checks, and error mapping require explicit positive and negative tests.
- Generated code does not need direct unit tests unless custom behavior is added around it.
- Mapping code must be tested when it contains conditions, transformations, defaults, or compatibility behavior.
- Do not chase coverage with meaningless tests. Add meaningful assertions or leave generated/trivial code excluded according to project policy.

## Anti-cheating rules

Never do the following:

- Assert only that a value is non-null when exact behavior is known.
- Mock the class or function under test instead of its dependencies.
- Test a copy of the implementation logic in the test.
- Use broad `any()` matchers for values that matter to behavior.
- Use `Thread.sleep` to make timing pass.
- Add `@Disabled`, `ignore`, assumptions, or conditional returns to avoid failing scenarios.
- Delete negative tests for validation/security/error cases.
- Replace integration tests with mocked unit tests without equivalent coverage.
- Reduce assertions after implementation changes.

## Test naming

Use behavior-focused names. Good examples:

- `returns report when request is valid`
- `rejects missing bearer token`
- `maps expired consent to unprocessable entity`
- `keeps optional openapi field nullable`

Avoid names that only repeat method names.

## Completion criteria

A change is not complete until:

1. Unit tests cover changed function behavior and parameters.
2. REST API behavior is covered by Cucumber scenarios when relevant.
3. Negative, edge, and error paths are covered.
4. Tests fail without the production change or with a realistic regression.
5. Static analysis and formatting checks pass.
