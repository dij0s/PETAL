import pytest
from typing import Optional
from pydantic import BaseModel, Field

from src.modelling.structured_output import RouterOutput
from src.modelling.utils import reduce_missing_attributes

@pytest.fixture
def dummy_router_output():
    def _make(intent=None, topic=None, location=None, aggregated_query=None, needs_clarification=None):
        return RouterOutput(
            intent=intent,
            topic=topic,
            location=location,
            aggregated_query=aggregated_query,
            needs_clarification=needs_clarification if needs_clarification is not None else True,
        )
    return _make

# test when nothing is missing
# hence user query is nonsense
def test_reduce_missing_attributes_none_missing(dummy_router_output):
    obj = dummy_router_output(intent="data_request", topic="solar", location="Sion", aggregated_query="Example query", needs_clarification=False)
    result = reduce_missing_attributes(obj)
    assert result is None

# test when some fields are missing
def test_reduce_missing_attributes_some_missing(dummy_router_output):
    obj = dummy_router_output(intent=None, topic="solar", location=None, aggregated_query="Example query", needs_clarification=True)
    result = reduce_missing_attributes(obj)
    assert result is not None
    assert "intent" in result or "location" in result

# test when all fields are missing
def test_reduce_missing_attributes_all_missing(dummy_router_output):
    obj = dummy_router_output()
    result = reduce_missing_attributes(obj)
    assert result is not None
    for field in ["intent", "topic", "location", "aggregated_query"]:
        assert field in result
