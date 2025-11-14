
import pytest
from utils import base62_encode, make_execution_id, create_timer
import time
from datetime import datetime

def test_base62_encode():
    assert base62_encode(0) == "0"
    assert base62_encode(10) == "A"
    assert base62_encode(61) == "z"
    assert base62_encode(62) == "10"
    assert base62_encode(12345) == "3d7"
    assert base62_encode(12345, limit=2) == "d7"

def test_make_execution_id():
    exec_id = make_execution_id()
    assert isinstance(exec_id, str)
    assert len(exec_id) > 0
    parts = exec_id.split('-')
    assert len(parts) == 2
    date_part, random_part = parts
    assert len(date_part) == 8
    assert len(random_part) == 5

def test_create_timer(mocker):
    mocker.patch('time.time', side_effect=[100.0, 100.5, 101.0, 102.0])
    
    # Test without laps
    timer = create_timer()
    # with pytest.capture_stdout() as captured:
    #     timer("first lap")
    #     assert "first lap: 0.5000" in captured.read()
    
    # Test with laps
    laps_timer = create_timer(is_laps=True)
    # with pytest.capture_stdout() as captured:
    #     laps_timer("second lap")
    #     assert "second lap: 0.5000" in captured.read()
    #     laps_timer("third lap")
    #     assert "third lap: 1.0000" in captured.read()
