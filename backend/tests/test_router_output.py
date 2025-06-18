import pytest
from pydantic import ValidationError
from src.modelling.structured_output import RouterOutput

def test_router_output_valid():
    # data for valid router output
    data = {
        "intent": "data_request",
        "location": "Sion",
        "needs_clarification": False
    }
    output = RouterOutput(**data)
    assert output.intent == "data_request"
    assert output.location == "Sion"
    assert output.needs_clarification is False

def test_router_output_optional_fields():
    # output with optional fields
    output = RouterOutput()
    assert output.intent is None
    assert output.location is None
    # needs_clarification may default to True
    # or None depending on model as it is
    # infered in the same prompt
    assert output.needs_clarification is True or output.needs_clarification is None

def test_router_output_invalid_type():
    # invalid types should raise ValidationError
    with pytest.raises(ValidationError):
        RouterOutput(intent=123, location=789, needs_clarification="yes")
