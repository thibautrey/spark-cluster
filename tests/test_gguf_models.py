import pytest

from spark_cluster.gguf_models import smallest_gguf


def test_variant_selection_sums_shards_and_keeps_shared_tables_and_projector():
    selection = smallest_gguf(
        (
            ("model-Q2-00001-of-00003.gguf", 1),
            ("model-Q2-00002-of-00003.gguf", 20),
            ("model-Q4-00001-of-00003.gguf", 1),
            ("model-Q4-00002-of-00003.gguf", 40),
            ("model-Engram-Q4.gguf", 30),
            ("mmproj-model.gguf", 2),
        )
    )
    assert selection.model_file == "model-Q2-00001-of-00003.gguf"
    assert selection.weight_bytes == 51
    assert "model-Q4-00002-of-00003.gguf" not in selection.files
    assert "mmproj-model.gguf" in selection.files
    assert selection.shared_shard_source == "model-Engram-Q4.gguf"
    assert selection.shared_shard_name == "model-Q2-00003-of-00003.gguf"


def test_missing_model_shards_are_not_replaced_by_an_unrelated_table():
    with pytest.raises(ValueError, match="missing model shards"):
        smallest_gguf((("model-Q2-00001-of-00004.gguf", 1), ("model-Engram.gguf", 2)))


def test_projector_alone_is_not_a_model():
    with pytest.raises(ValueError, match="No GGUF model"):
        smallest_gguf((("mmproj-model.gguf", 1),))
