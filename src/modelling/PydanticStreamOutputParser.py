"""Parser for Pydantic output."""

from __future__ import annotations

import json
from typing import Any, Optional, TypeVar, Union, Type, List

import jsonpatch
import pydantic

from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import (
    parse_json_markdown,
)
from langchain_core.utils.pydantic import IS_PYDANTIC_V1

if IS_PYDANTIC_V1:
    PydanticBaseModel = pydantic.BaseModel
else:
    from pydantic.v1 import BaseModel
    PydanticBaseModel = Union[BaseModel, pydantic.BaseModel]

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)

_PYDANTIC_STREAM_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

class PydanticStreamOutputParser(BaseCumulativeTransformOutputParser[TBaseModel]):
    """Parse the output of an LLM call to a Pydantic object.

    When used in streaming mode, it will yield partial JSON objects containing
    all the keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields JSONPatch operations
    describing the difference between the previous and the current object.

    This parser was proposed as an enhancement by @YanSte in the open-source langchain framework as per the following discussion: https://github.com/langchain-ai/langchain/discussions/19225.
    The parser has been modified to handle cases where values can optionally be null.
    """

    pydantic_object: Type[TBaseModel]

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def _get_schema(self, pydantic_object: type[TBaseModel]) -> Optional[dict[str, Any]]:
        if issubclass(pydantic_object, pydantic.BaseModel):
            return pydantic_object.model_json_schema()
        if issubclass(pydantic_object, pydantic.v1.BaseModel):
            return pydantic_object.schema()
        return None

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        try:
            json_object = parse_json_markdown(text)
            result = self.pydantic_object.parse_obj(json_object)
            return result
        except json.JSONDecodeError:
            return None
        except pydantic.ValidationError:
            return None

    def parse(self, text: str) -> TBaseModel:
        return self.parse_result([Generation(text=text)])

    @property
    def _type(self) -> str:
        return "pydantic_stream_output_parser"

    @property
    def OutputType(self) -> Type[TBaseModel]:
        """Return the pydantic model."""
        return self.pydantic_object

    def get_format_instructions(self) -> str:
        schema = self._get_schema(self.pydantic_object)

        if not schema:
            return "Return a JSON object."

        reduced_schema = dict(schema.items())
        # Remove extraneous fields.
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        return _PYDANTIC_STREAM_FORMAT_INSTRUCTIONS.format(schema=schema_str)
