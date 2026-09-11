"""LangChain's own conformance suite, run against all three tools.

These are not our tests — they are `langchain-tests`, the suite LangChain
publishes so an integration can be checked against langchain-core's interfaces
by anyone, not just by its author. Every tool here runs both the unit and the
integration suite.

The integration suite invokes the tools for real. That is deliberate: it hits
the open browse route, which needs no API key, so this passes in CI with no
secret configured.
"""
from typing import Type

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_greencalculus import (
    EmissionFactorLookup,
    EmissionFactorSearch,
    EmissionsCalculator,
)

UK_GRID = "grid.gbr.electricity.location_based"


class _Lookup:
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return EmissionFactorLookup

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"key": UK_GRID}


class TestLookupUnit(_Lookup, ToolsUnitTests): ...
class TestLookupIntegration(_Lookup, ToolsIntegrationTests): ...


class _Search:
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return EmissionFactorSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "UK grid electricity", "limit": 3}


class TestSearchUnit(_Search, ToolsUnitTests): ...
class TestSearchIntegration(_Search, ToolsIntegrationTests): ...


class _Calc:
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return EmissionsCalculator

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        # No API key in CI, so this exercises the documented keyless path: the
        # multiplication is done locally and the tool SAYS it carries no
        # server-side audit trail rather than implying it does.
        return {"key": UK_GRID, "quantity": 1000}


class TestCalcUnit(_Calc, ToolsUnitTests): ...
class TestCalcIntegration(_Calc, ToolsIntegrationTests): ...
