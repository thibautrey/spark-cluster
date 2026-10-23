from spark_cluster.catalog_cache import CatalogCache


def test_catalog_cache_survives_reopen_and_separates_metadata_from_fits(tmp_path):
    path = tmp_path / "catalog-cache.db"
    cache = CatalogCache(path)
    cache.set_metadata("metadata-key", {"id": "acme/model"}, "next-page", 60)
    cache.set_fit("fit-key", {"repo_id": "acme/model", "compatible": True}, 60)
    cache.close()

    reopened = CatalogCache(path)
    assert reopened.get_metadata("metadata-key") == ({"id": "acme/model"}, "next-page")
    assert reopened.get_fit("fit-key") == {"repo_id": "acme/model", "compatible": True}
    assert reopened.stats() == {"metadata_entries": 1, "fit_entries": 1}
    reopened.close()
