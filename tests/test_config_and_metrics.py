import pytest
from pydantic import ValidationError

from app.config import Settings
from app.metrics import percentile


def test_placeholder_jwt_secret_is_rejected():
    with pytest.raises(ValidationError):
        Settings(jwt_secret="change-me-to-a-long-random-string", _env_file=None)


def test_short_jwt_secret_is_rejected():
    with pytest.raises(ValidationError):
        Settings(jwt_secret="short", _env_file=None)


def test_empty_jwt_secret_falls_back_to_random():
    s = Settings(jwt_secret="", _env_file=None)
    assert len(s.effective_jwt_secret()) >= 32


def test_percentile_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 95) == 95
    assert percentile(values, 50) == 50
    assert percentile([42.0], 95) == 42.0
