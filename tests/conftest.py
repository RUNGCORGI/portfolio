import pytest
from core.fixtures import load_sample
from core.portfolio import build_portfolio


@pytest.fixture(scope="session")
def md():
    return load_sample()


@pytest.fixture(scope="session")
def pf(md):
    return build_portfolio(md, [1, 1, 1])
