"""What the conformance suite does not check: that the output is actually useful.

langchain-tests proves the tool matches langchain-core's interface. It says
nothing about whether the string handed back to a model is any good — and for a
tool whose entire purpose is carrying provenance, that is the part that matters.
"""
import os

import pytest

from langchain_greencalculus import (
    EmissionFactorLookup,
    EmissionFactorSearch,
    EmissionsCalculator,
)

UK_GRID = "grid.gbr.electricity.location_based"


def test_lookup_always_carries_a_citation():
    """The reason this package exists. A bare number is the failure mode."""
    out = EmissionFactorLookup().invoke({"key": UK_GRID})
    assert "citation:" in out
    assert "DEFRA" in out or "Department for Energy" in out or "DESNZ" in out
    assert "kg CO2e" in out


def test_lookup_returns_the_published_value():
    out = EmissionFactorLookup().invoke({"key": UK_GRID})
    assert out.startswith("0.13096 ")


def test_unknown_key_tells_the_model_what_to_do_next():
    """A stack trace or a bare 404 leaves an agent stuck in a retry loop."""
    out = EmissionFactorLookup().invoke({"key": "not.a.real.factor.key"})
    assert "search_emission_factors" in out
    assert "Traceback" not in out


def test_search_returns_keys_a_model_can_feed_back_in():
    out = EmissionFactorSearch().invoke({"query": "UK grid electricity", "limit": 3})
    assert "lookup_emission_factor" in out       # tells the model the next step
    assert "grid." in out                         # real keys, not prose


def test_search_miss_suggests_a_fix_rather_than_failing_silently():
    out = EmissionFactorSearch().invoke({"query": "zzzz nonexistent zzzz", "limit": 3})
    assert "Nothing matched" in out or "candidate" in out


def test_keyless_calculate_admits_it_has_no_audit_trail():
    """Silently multiplying and implying a traced result would be the worst
    possible behaviour for a tool sold on auditability."""
    tool = EmissionsCalculator(api_key=None)
    if tool.api_key:                      # a key is present in this environment
        pytest.skip("GREENCALCULUS_API_KEY set; keyless path not exercised")
    out = tool.invoke({"key": UK_GRID, "quantity": 1000})
    assert "130.96" in out
    assert "no server-side audit trail" in out
    assert "citation:" in out


def test_api_key_is_picked_up_from_the_environment(monkeypatch):
    monkeypatch.setenv("GREENCALCULUS_API_KEY", "gc_live_dummy")
    assert EmissionFactorLookup().api_key == "gc_live_dummy"


def test_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("GREENCALCULUS_API_KEY", "gc_live_env")
    assert EmissionFactorLookup(api_key="gc_live_explicit").api_key == "gc_live_explicit"


def test_descriptions_tell_the_model_to_prefer_the_tool_over_recall():
    """The description is the interface an LLM reads. If it does not say
    'use this instead of remembering', the model will remember instead."""
    d = EmissionFactorLookup().description.lower()
    assert "instead of" in d and "memory" in d
    assert "citation" in d


def test_search_description_warns_about_the_basis_trap():
    """location-based vs market-based is the mistake that silently produces a
    wrong-but-plausible number, so the tool must name it."""
    d = EmissionFactorSearch().description.lower()
    assert "location-based" in d and "market-based" in d


@pytest.mark.asyncio
async def test_async_path_returns_the_same_answer():
    tool = EmissionFactorLookup()
    assert await tool.ainvoke({"key": UK_GRID}) == tool.invoke({"key": UK_GRID})


@pytest.mark.skipif(not os.environ.get("GREENCALCULUS_API_KEY"),
                    reason="keyed calculation needs GREENCALCULUS_API_KEY")
def test_keyed_calculate_returns_working_citation_and_receipt():
    """With a key the point is the audit trail, so assert the trail, not the number."""
    out = EmissionsCalculator().invoke(
        {"key": UK_GRID, "quantity": 12000})
    assert "1571.52" in out
    assert "working:" in out and "×" in out          # the arithmetic is shown
    assert "citation:" in out and "E25" in out        # the exact cell
    assert "receipt:" in out                          # reproducibility hash
    assert "verify.greencalculus.com" in out          # checkable without an account
    assert "scope2" in out                            # GHG Protocol mapping


def test_calculate_infers_the_unit_when_the_model_omits_it():
    """Agents routinely omit optional args; a 400 would end the run."""
    out = EmissionsCalculator().invoke({"key": UK_GRID, "quantity": 1})
    assert "kWh" in out
    assert "Calculation failed" not in out
