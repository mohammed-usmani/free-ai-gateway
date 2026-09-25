import litellm
import pytest

from tests import fake_upstream

litellm.suppress_debug_info = True


@pytest.fixture(scope="session")
def upstream_url():
    return fake_upstream.start()


@pytest.fixture(autouse=True)
def _clear_calls():
    fake_upstream.calls.clear()
