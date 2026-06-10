"""Shared fixtures for test isolation."""

import pytest

from lucide import search


@pytest.fixture(autouse=True)
def isolated_search_cache(tmp_path, monkeypatch):
    """Isolate tests from the real search-DB cache and the network.

    Since the v0.3.0 release published the search-DB asset, the download
    fallback in ``_resolve_search_db`` succeeds for real. Without isolation,
    tests that reach it fetch the DB into ``~/.cache/python-lucide`` and
    pollute later "no db" tests.
    """
    cache_dir = tmp_path / "search-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(search, "_get_cache_dir", lambda: cache_dir)

    def _no_network(url, *_args, **_kwargs):
        raise AssertionError(f"unexpected download attempt: {url}")

    monkeypatch.setattr(search.urllib.request, "urlretrieve", _no_network)
