# Technical Overview
- Kotlin 2.3 - Programming Language
- jUnit 6, kotlin.test - Unit Testing

# Kotlin
- Use always named arguments for function calls
- Avoid Single-expression functions

# Testing
- Use JUnit 5 for unit tests
- Use `kotlin.test` for assertions and annotations if possible

# Database
- Use only sequences for identifiers in entities
- Liquibase scripts must have preconditions to avoid errors. Multiple changes require multiple changelogs (e.g. table creation, constraints)
- Use only QueryDSL for database queries.
