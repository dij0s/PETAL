import pytest
from src.modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from src.modelling.structured_output import RouterOutput

# fixture for parser
@pytest.fixture
def parser():
    # create parser with routeroutput
    return PydanticStreamOutputParser(pydantic_object=RouterOutput)

def test_pydantic_stream_output_parser_parse_valid(parser):
    # valid json string
    json_str = '{"intent": "planning_request", "topic": "wind", "location": "Sion", "needs_clarification": false}'
    result = parser.parse(json_str)
    # check type
    assert isinstance(result, RouterOutput)
    # check fields
    assert result.intent == "planning_request"
    assert result.topic == "wind"
    assert result.location == "Sion"
    assert result.needs_clarification is False

def test_pydantic_stream_output_parser_parse_invalid_json(parser):
    # invalid json (missing closing brace)
    invalid_json = '{"intent": "policy_question", "topic": "biomass", "location": "Sion", "needs_clarification": false'
    result = parser.parse(invalid_json)
    # accept none or valid routeroutput
    if result is not None:
        assert isinstance(result, RouterOutput)
        assert result.intent == "policy_question"
        assert result.topic == "biomass"
        assert result.location == "Sion"
        assert result.needs_clarification is False
    else:
        assert result is None

def test_pydantic_stream_output_parser_parse_invalid_schema(parser):
    # bad types in json
    bad_json = '{"intent": 123, "topic": 456, "location": 789, "needs_clarification": "maybe"}'
    result = parser.parse(bad_json)
    # should return none
    assert result is None

def test_pydantic_stream_output_parser_format_instructions(parser):
    # get format instructions
    instructions = parser.get_format_instructions()
    # check for key phrases
    assert "output should be formatted as a JSON instance" in instructions
    assert "output schema" in instructions

def test_pydantic_stream_output_parser_description(parser):
    # get description
    desc = parser.get_description()
    # check for fields
    assert "-intent:" in desc
    assert "-topic:" in desc
    assert "-location:" in desc
    assert "-needs_clarification:" in desc
