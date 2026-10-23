import pytest

from spark_cluster.spark_optimization import classify_spark


@pytest.mark.parametrize(
    "model",
    [
        {"id": "nvidia/ordinary-model", "tags": ["blackwell", "fp4", "modelopt", "mtp"]},
        {"id": "acme/MXFP4-model", "tags": ["mxfp4"]},
        {"id": "acme/GGUF-Q4_K_M", "tags": ["gguf"]},
        {"id": "nvfp4/generic", "tags": ["base_model:nvidia/model-NVFP4"]},
        {"id": "acme/myNVFP4ish-model", "tags": ["not-nvfp4"]},
        {"id": "acme/Not-NVFP4-model"},
        {"id": "acme/spark-model", "tags": ["spark", "sm100"]},
    ],
)
def test_generic_formats_and_base_model_names_do_not_claim_spark_optimization(model):
    assert classify_spark(model).score == 0


def test_exact_spark_target_outranks_vendor_nvfp4_candidate():
    targeted = classify_spark({"id": "acme/Qwen-NVFP4-DGX-Spark", "tags": ["nvfp4"]})
    nvidia = classify_spark({"id": "nvidia/Qwen-NVFP4"})
    assert targeted.score > nvidia.score > 0
    assert targeted.tier == "spark-targeted"
    assert "NVFP4" in targeted.badges
    assert any("not a measured speedup" in note for note in targeted.caveats)
    assert nvidia.tier == "nvfp4"


def test_quantization_metadata_is_stronger_than_repository_name():
    metadata = classify_spark(
        {
            "id": "acme/model",
            "config": {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}},
        }
    )
    name = classify_spark({"id": "acme/model-NVFP4"})
    assert metadata.score > name.score > 0
    assert "quantization metadata" in metadata.reasons[0]


def test_nvfp4_gguf_has_engine_warning_and_does_not_claim_vllm_support():
    result = classify_spark({"id": "acme/model-NVFP4-GGUF", "tags": ["nvfp4", "gguf"]})
    assert result.score > 0
    assert any("llama.cpp" in note for note in result.caveats)


def test_datacenter_only_and_conflicting_quantization_do_not_get_nvfp4_boost():
    other = classify_spark({"id": "acme/model-NVFP4", "tags": ["sm100-only"]})
    assert other.score == 0
    assert other.tier == "other-hardware"
    conflict = classify_spark(
        {"id": "acme/model-NVFP4", "config": {"quantization_config": {"quant_method": "gptq"}}}
    )
    assert conflict.score == 0
    assert any("conflicts" in note for note in conflict.caveats)


def test_nvidia_namespace_is_not_taken_from_untrusted_author_field():
    result = classify_spark({"id": "acme/model-NVFP4", "author": "nvidia"})
    assert "NVIDIA checkpoint" not in result.badges


def test_deepinfer_and_sm100_are_not_sm121_compatibility_evidence():
    result = classify_spark({"id": "acme/model", "library_name": "DeepInfer"})
    assert result.score == 0
    assert any("does not establish SM121" in note for note in result.caveats)


def test_recipe_with_no_checkpoint_files_does_not_outrank_downloadable_models():
    result = classify_spark(
        {
            "id": "acme/Model-NVFP4-GB10-Recipe",
            "tags": ["dgx-spark", "nvfp4"],
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "run.sh"}],
        }
    )
    assert result.score == 0
    assert result.tier == "metadata-only"


def test_missing_optional_metadata_is_supported():
    assert classify_spark({"id": "acme/model", "tags": None, "library_name": None}).score == 0
