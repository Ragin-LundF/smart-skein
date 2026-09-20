# Test fixtures

Two committed files let the integration tests run against **real** ONNX Runtime and the **real**
Hugging Face tokenizer binding, rather than mocks. A genuine sentence encoder is 20–400 MB and
cannot go in a repository, so both are the smallest artifacts with the same interface.

| File | Size | Provenance |
|---|---|---|
| `tiny-embedding-model.onnx` | 317 B | **Generated** by [`tiny_embedding_model.py`](tiny_embedding_model.py) — run `python3 tiny_embedding_model.py`, standard library only |
| `tiny-tokenizer.json` | 548 B | **Hand-authored.** No generator, and none is wanted: it is a literal `tokenizers` config with a six-token `WordLevel` vocabulary, and a script emitting it would be longer than the file |

## What they prove, and what they do not

The `Gather` lookup in the ONNX file is not a toy stand-in for a real architecture — it *is* a
static embedding model, the Model2Vec-class approach. So the adapter's interface, pooling, masking,
batching, caching and fingerprinting are all exercised for real.

What they do **not** prove is that a 384-dimension MiniLM behaves as expected end to end. A
transformer differs in what fills the embedding table, not in the interface this adapter talks to,
but that is an argument rather than a test. Worth one smoke test against a genuinely exported model
before relying on this in production.
