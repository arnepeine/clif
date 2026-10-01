from __future__ import annotations

from pathlib import Path

import pytest

from clif.config import Settings
from clif.devtools.demo_release import write_demo_release
from clif.indexing.build import build_index
from clif.models.schemas import MapRequest
from clif.pipeline import ClifMapper
from clif.reranking import HeuristicReranker
from clif.terminology.rf2 import import_packages


@pytest.fixture(scope="session")
def demo_package(tmp_path_factory) -> Path:
    return write_demo_release(tmp_path_factory.mktemp("rf2"))


@pytest.fixture(scope="session")
def settings(tmp_path_factory, demo_package) -> Settings:
    s = Settings(data_dir=tmp_path_factory.mktemp("data"), _env_file=None)
    import_packages([demo_package], s.db_path, reset=True)
    build_index(s)
    return s


@pytest.fixture(scope="session")
def mapper(settings) -> ClifMapper:
    return ClifMapper.from_settings(settings, reranker=HeuristicReranker())


@pytest.fixture
def map_text(mapper):
    def _map(text: str, **kw):
        return mapper.map(MapRequest(text=text, **kw))

    return _map
