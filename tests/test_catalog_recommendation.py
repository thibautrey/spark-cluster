from datetime import date

from spark_cluster.catalog import summary
from spark_cluster.catalog_recommendation import REPORTS, ranking_key, recommend
from spark_cluster.spark_optimization import classify_spark


def test_revision_and_expiry_bound_community_evidence():
    repo, report = next(iter(REPORTS.items()))
    data = {"id": repo, "sha": report["commit"]}
    spark = classify_spark(data)
    current = recommend(data, spark, date(2026, 10, 23))
    assert current.priority == 2
    assert current.reported_nodes == [2]
    assert current.sources
    assert "Community" in " ".join(current.caveats)
    assert recommend(data, spark, date(2026, 11, 24)).priority < 2
    assert recommend({**data, "sha": "new"}, spark, date(2026, 10, 23)).priority < 2
    assert recommend({**data, "siblings": [{"rfilename": "README.md"}]},
                     classify_spark({**data, "siblings": [{"rfilename": "README.md"}]}),
                     date(2026, 10, 23)).priority == -1


def test_modified_experimental_and_other_hardware_are_explicit():
    for repo, library, variant in [
        ("community/Qwen-Uncensored-GGUF", "gguf", "modified"),
        ("community/Qwen-abliterated", "transformers", "modified"),
        ("lmstudio-community/Qwen-MLX-4bit", "transformers", "other-hardware"),
        ("Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF", "gguf", "experimental"),
    ]:
        model = summary({"id": repo, "library_name": library})
        assert model.recommendation.variant == variant
        assert model.recommendation.priority == -1
        assert model.recommendation.caveats


def test_popularity_never_overrides_evidence_and_order_is_stable():
    models = [summary(data) for data in [
        {"id": "a/popular", "downloads": 10**12, "likes": 10**9},
        {"id": "z/candidate", "tags": ["dgx-spark"]},
        {"id": "b/same"}, {"id": "a/same"},
    ]]
    ordered = sorted(models, key=ranking_key)
    assert ordered[0].id == "z/candidate"
    assert sorted(reversed(models), key=ranking_key) == ordered
    assert ordered[-2].id == "a/same"


def test_publisher_identity_is_not_inferred_from_base_model_or_substring():
    assert summary({"id": "Qwen/Qwen3.8-27B"}).recommendation.publisher == "original"
    assert summary({"id": "nvidia/MiniMax-M3-NVFP4"}).recommendation.publisher == "vendor-conversion"
    assert summary({"id": "not-qwen/Qwen3.8", "cardData": {
        "base_model": "Qwen/Qwen3.8-27B"}}).recommendation.publisher == "community"
