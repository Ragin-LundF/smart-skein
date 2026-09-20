"""Generates tiny-embedding-model.onnx, a real ONNX model small enough to commit.

Why this exists
---------------
`OnnxEmbeddingRuntimeIntegrationTest` needs a genuine ONNX file so the ONNX Runtime wiring
-- session creation, input binding, shape handling, output extraction -- is exercised rather
than mocked. A real sentence encoder is 20-400 MB, which cannot go in a repository, so this
builds the smallest model with the same interface: a token-embedding lookup.

    input_ids [batch, seq] int64  ->  Gather(embeddings, axis=0)  ->  [batch, seq, hidden] float

That is not a toy stand-in for a real architecture -- it *is* a static embedding model, the
Model2Vec-class approach that section 04 of the plan identifies as the practical middle
ground between n-grams and a transformer. A transformer differs in what fills the embedding
table, not in the interface this adapter talks to.

The ONNX wire format is protobuf, so this writes it directly: `onnx` and `torch` are not
available here, and adding a build-time Python ML dependency to a Kotlin repository to emit
600 bytes would be a poor trade.

Run:  python3 tiny_embedding_model.py
Needs only the standard library.
"""

import struct

VOCAB = 6
HIDDEN = 4

FLOAT = 1
INT64 = 7
ATTRIBUTE_INT = 2


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def tag(field, wire):
    return varint((field << 3) | wire)


def var_field(field, value):
    return tag(field, 0) + varint(value)


def len_field(field, payload):
    return tag(field, 2) + varint(len(payload)) + payload


def string_field(field, text):
    return len_field(field, text.encode("utf-8"))


def dimension(value):
    """TensorShapeProto.Dimension: dim_value = 1 (int64) or dim_param = 2 (string)."""
    if isinstance(value, int):
        return var_field(1, value)
    return string_field(2, value)


def tensor_type(elem_type, shape):
    dims = b"".join(len_field(1, dimension(d)) for d in shape)
    tensor = var_field(1, elem_type) + len_field(2, dims)
    return len_field(1, tensor)                      # TypeProto.tensor_type = 1


def value_info(name, elem_type, shape):
    return string_field(1, name) + len_field(2, tensor_type(elem_type, shape))


def embedding_table():
    """A readable table: row r is [r, r + 0.5, -r, 1.0], so a pooled result is checkable by hand."""
    values = []
    for row in range(VOCAB):
        values.extend([float(row), row + 0.5, -float(row), 1.0])
    raw = b"".join(struct.pack("<f", v) for v in values)
    dims = var_field(1, VOCAB) + var_field(1, HIDDEN)
    return dims + var_field(2, FLOAT) + string_field(8, "embeddings") + len_field(9, raw)


def gather_node():
    axis = string_field(1, "axis") + var_field(3, 0) + var_field(20, ATTRIBUTE_INT)
    return (
        string_field(1, "embeddings")
        + string_field(1, "input_ids")
        + string_field(2, "last_hidden_state")
        + string_field(3, "lookup")
        + string_field(4, "Gather")
        + len_field(5, axis)
    )


def graph():
    return (
        len_field(1, gather_node())
        + string_field(2, "tiny_embedding")
        + len_field(5, embedding_table())
        + len_field(11, value_info("input_ids", INT64, ["batch", "seq"]))
        + len_field(12, value_info("last_hidden_state", FLOAT, ["batch", "seq", HIDDEN]))
    )


def model():
    opset = string_field(1, "") + var_field(2, 13)
    return (
        var_field(1, 8)                               # ir_version
        + string_field(2, "smart-skein")              # producer_name
        + len_field(7, graph())
        + len_field(8, opset)
    )


if __name__ == "__main__":
    payload = model()
    with open("tiny-embedding-model.onnx", "wb") as handle:
        handle.write(payload)
    print(f"wrote tiny-embedding-model.onnx ({len(payload)} bytes)")
