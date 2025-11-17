
import pytest
from utils import base62_encode, make_execution_id, create_timer
import time
from datetime import datetime
from config import from_dict, ModelConfig, get_config


def test_from_dict():
    a = {"n_head": 1, "not_involved": 2}
    b = from_dict(ModelConfig, a)
    assert b.n_head == 1
    with pytest.raises(AttributeError):
        b.not_involved

def test_from_dict_with_override():
    a = {"n_head": 1, "not_involved": 2}
    b = from_dict(ModelConfig, a, override={"n_head": 2})
    assert b.n_head == 2

def test_get_test_config():
    config = get_config(preset="test")
    assert config.preset == "test"
    assert config.max_steps == 1000
