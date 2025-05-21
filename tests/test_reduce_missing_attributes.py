import pytest
from src.modelling.utils import reduce_missing_attributes

# dummy class for testing
class DummyRouterOutput:
    def __init__(self, intent=None, topic=None, location=None, needs_clarification=None):
        self.intent = intent
        self.topic = topic
        self.location = location
        self.needs_clarification = needs_clarification
    def model_dump(self):
        # returns dict of attributes
        return {
            "intent": self.intent,
            "topic": self.topic,
            "location": self.location,
            "needs_clarification": self.needs_clarification,
        }

@pytest.fixture
def dummy_router_output():
    def _make(intent=None, topic=None, location=None, needs_clarification=None):
        return DummyRouterOutput(
            intent=intent,
            topic=topic,
            location=location,
            needs_clarification=needs_clarification,
        )
    return _make

# test when nothing is missing
# hence user query is nonsense
def test_reduce_missing_attributes_none_missing(dummy_router_output):
    obj = dummy_router_output(intent="data_request", topic="solar", location="Sion", needs_clarification=False)
    result = reduce_missing_attributes(obj)
    assert result is None

# test when some fields are missing
def test_reduce_missing_attributes_some_missing(dummy_router_output):
    obj = dummy_router_output(intent=None, topic="solar", location=None, needs_clarification=True)
    result = reduce_missing_attributes(obj)
    assert result is not None
    assert "intent" in result or "location" in result

# test when all fields are missing
def test_reduce_missing_attributes_all_missing(dummy_router_output):
    obj = dummy_router_output()
    result = reduce_missing_attributes(obj)
    assert result is not None
    for field in ["intent", "topic", "location", "needs_clarification"]:
        assert field in result
