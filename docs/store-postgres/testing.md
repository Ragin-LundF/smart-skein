# Testing


Integration tests use **Testcontainers** with `postgres:16-alpine` and therefore **require Docker**.
They cover:

- single-insert and ordered read-back round-trips (indices/values/labels preserved),
- `clear()` emptying the table,
- `addAll` batch insert,
- **encrypted round-trip while verifying the at-rest blob is genuinely ciphertext** (the stored bytes
  differ from the plaintext codec output under `AesGcmEncryption`).

Unit tests (no Docker) cover the codec round-trip and the AES-GCM round-trip + tamper detection.

```bash
./gradlew :skein-store-postgres:test          # needs Docker for the integration tests
```

---

[PostgreSQL storage](README.md) · [Components](components.md) · [Testing](testing.md) · [All documentation](../README.md)
