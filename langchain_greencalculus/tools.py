"""LangChain tools for sourced greenhouse-gas emission factors.

Design note, because it is the whole point of this package:

An LLM asked for an emission factor from memory is right about 46% of the time,
and names the correct publisher while giving the wrong number in up to 59% of
answers — a failure that survives review, because a wrong value wearing DEFRA's
name looks like a right one. Given a lookup tool, the same models reach 99%.
Those figures are measured, not asserted: https://doi.org/10.5281/zenodo.22692277

So every tool here returns **the citation alongside the number**, and the tool
descriptions instruct the model to repeat it. A number without its source is the
thing this package exists to stop.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

try:  # pragma: no cover - import shape only
    from greencalculus import GreenCalculus, GreenCalculusError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-greencalculus requires the greencalculus SDK: pip install greencalculus"
    ) from exc


def _client(api_key: Optional[str]) -> GreenCalculus:
    # No key is a supported mode, not a degraded one: the corpus browse route is
    # open, so lookup and search work with nothing to sign up for. Only
    # calculations are gated.
    return GreenCalculus(api_key=api_key) if api_key else GreenCalculus(api_key=None)


def _citation(row: Dict[str, Any]) -> str:
    """The publisher's own citation string, printed verbatim.

    Never re-assemble this from parts. The API returns a canonical `citation.text`
    precisely so that every surface prints the same string and it cannot drift.
    """
    factor = row.get("factor") or row
    cit = factor.get("citation") or {}
    text = cit.get("text")
    if text:
        return text
    src = factor.get("source") or row.get("source") or {}
    bits = [src.get("id"), src.get("cell_ref")]
    return ", ".join(b for b in bits if b) or "source not returned"


class _Base(BaseTool):
    """Shared client handling. `api_key` is optional throughout.

    Falls back to GREENCALCULUS_API_KEY, and works with neither — the browse
    route is open, so an agent can be wired up before anyone has signed up.
    """

    api_key: Optional[str] = None
    _gc: Optional[GreenCalculus] = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        if not kwargs.get("api_key"):
            env = os.environ.get("GREENCALCULUS_API_KEY")
            if env:
                kwargs["api_key"] = env
        super().__init__(**kwargs)

    @property
    def gc(self) -> GreenCalculus:
        if self._gc is None:
            self._gc = _client(self.api_key)
        return self._gc

    async def _arun(self, *args: Any,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
                    **kwargs: Any) -> str:
        """The SDK is synchronous, so offload rather than block the event loop.

        A naive async wrapper that just called _run would stall every other
        coroutine in an agent for the duration of the HTTP request.
        """
        return await asyncio.to_thread(self._run, *args, **kwargs)


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #
class LookupInput(BaseModel):
    key: str = Field(
        description=(
            "The exact factor key, e.g. 'grid.gbr.electricity.location_based' or "
            "'fuels.gbr.diesel_average_biofuel_blend.litre'. If you do not already "
            "know the key, call search_emission_factors first — do not guess a key."
        )
    )
    as_of: Optional[str] = Field(
        default=None,
        description=(
            "Optional data version or ISO date to pin the answer to, e.g. '2026.187'. "
            "Use when a figure must reproduce exactly later, such as for a filed "
            "disclosure. Requires an API key."
        ),
    )


class EmissionFactorLookup(_Base):
    """Look up one emission factor by its exact key."""

    name: str = "lookup_emission_factor"
    description: str = (
        "Get a greenhouse-gas emission factor by its exact key, with the source it "
        "came from. Returns the value, the unit, and a citation naming the publisher "
        "and the exact cell in their workbook. "
        "ALWAYS use this instead of recalling a factor from memory: unaided recall is "
        "correct about 46% of the time and frequently pairs the right publisher with "
        "the wrong number. ALWAYS repeat the citation to the user alongside the value."
    )
    args_schema: Type[BaseModel] = LookupInput

    def _run(self, key: str, as_of: Optional[str] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            row = self.gc.factor(key, as_of=as_of) if as_of else self.gc.factor(key)
        except GreenCalculusError as e:
            if e.status == 401 and as_of:
                return ("Pinning to a data version needs an API key (free at "
                        "https://greencalculus.com/developers/). Retry without as_of "
                        "for the current value.")
            if e.status == 404:
                return (f"No factor with key '{key}'. Call search_emission_factors "
                        f"with a plain-English description instead of guessing keys.")
            return f"Lookup failed ({e.code}): {e.message}"
        return (f"{row['value']} {row['unit']}\n"
                f"key: {key}\n"
                f"citation: {_citation(row)}")


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
class SearchInput(BaseModel):
    query: str = Field(
        description="Plain-English description, e.g. 'UK grid electricity' or 'diesel per litre'."
    )
    limit: int = Field(default=5, description="How many candidates to return (1-25).")


class EmissionFactorSearch(_Base):
    """Find candidate factors from a plain-English description."""

    name: str = "search_emission_factors"
    description: str = (
        "Search the emission-factor corpus in plain English and get back candidate "
        "keys with their values, units and publishers. Use this FIRST whenever you do "
        "not already know the exact key, then call lookup_emission_factor. "
        "Choosing between candidates matters: factors differ by country, by year, and "
        "by basis (location-based vs market-based, well-to-tank vs combustion). If the "
        "user has not said which they need, ask rather than picking one silently."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, limit: int = 5,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            res = self.gc.search(query, limit=max(1, min(limit, 25)))
        except GreenCalculusError as e:
            return f"Search failed ({e.code}): {e.message}"
        rows: List[Dict[str, Any]] = res.get("factors") or []
        if not rows:
            return (f"Nothing matched '{query}'. Try fewer words, or name the country "
                    f"and the unit, e.g. 'Germany grid electricity kWh'.")
        out = [f"{len(rows)} candidate(s) for '{query}':"]
        for r in rows:
            f = r.get("factor") or {}
            src = (r.get("source") or {}).get("id", "")
            out.append(f"- {r.get('key')} = {f.get('value')} {f.get('unit')}"
                       f"{f'  [{src}]' if src else ''}\n  {r.get('name','')}")
        out.append("Call lookup_emission_factor with the chosen key to get its citation.")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# calculate
# --------------------------------------------------------------------------- #
class CalculateInput(BaseModel):
    key: str = Field(description="The exact factor key to apply.")
    quantity: float = Field(description="How much activity, in the factor's own unit.")
    unit: Optional[str] = Field(
        default=None,
        description="The unit of `quantity`, if it may differ from the factor's unit.",
    )


class EmissionsCalculator(_Base):
    """Multiply an activity quantity by a sourced factor, keeping the audit trail."""

    name: str = "calculate_emissions"
    description: str = (
        "Multiply an activity quantity by a sourced emission factor and return the "
        "result with its full audit trail. Prefer this over doing the arithmetic "
        "yourself, because the returned trail records which factor and which data "
        "version produced the number — which is what makes the figure defensible "
        "later. Needs a free API key."
    )
    args_schema: Type[BaseModel] = CalculateInput

    def _run(self, key: str, quantity: float, unit: Optional[str] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key:
            # Fail honestly and usefully rather than silently multiplying: the
            # audit trail is the reason to call this at all.
            try:
                row = self.gc.factor(key)
            except GreenCalculusError as e:
                return f"Lookup failed ({e.code}): {e.message}"
            total = quantity * float(row["value"])
            return (f"{total:g} (derived locally: {quantity} x {row['value']} {row['unit']})\n"
                    f"citation: {_citation(row)}\n"
                    f"NOTE: no API key set, so this multiplication was done here and carries "
                    f"no server-side audit trail. A free key at "
                    f"https://greencalculus.com/developers/ returns a traced calculation.")
        # Shape per /v1/calculate/ghg-activity: {"activity": {value, unit},
        # "factor_key": "..."}. The unit defaults to the factor's own, so an
        # agent that omits it still gets a correct answer rather than a 400.
        if not unit:
            try:
                unit = self.gc.factor(key)["unit"].split(" per ")[-1]
            except (GreenCalculusError, KeyError, IndexError):
                return ("Could not determine the unit for this factor. Pass `unit` "
                        "explicitly, e.g. unit='kWh'.")
        try:
            res = self.gc.ghg_activity(activity={"value": quantity, "unit": unit},
                                       factor_key=key)
        except GreenCalculusError as e:
            return f"Calculation failed ({e.code}): {e.message}"
        # The raw response is rich (receipt hash, scope mapping, attribution).
        # An LLM does not need all of it and reads a wall of JSON badly, so
        # return the facts a model should repeat, and nothing else.
        em = res.get("emissions") or {}
        work = (res.get("working") or {}).get("formula", "")
        cit = _citation(res)
        proof = (res.get("proof_urls") or [""])[0]
        ver = (res.get("meta") or {}).get("gc_version", "n/a")
        rid = (res.get("receipt") or {}).get("id", "")
        lines = [f"{em.get('value')} {em.get('unit', 'kg CO2e')}"]
        if work:
            lines.append(f"working: {work}")
        scope = (res.get("scope") or {}).get("ghg_protocol")
        if scope:
            lines.append(f"scope: {scope}")
        lines.append(f"citation: {cit}")
        if res.get("note"):
            lines.append(f"note: {res['note']}")
        lines.append(f"data version: {ver}"
                     + (f"  receipt: {rid[:16]}…" if rid else "")
                     + (f"\nproof: {proof}" if proof else ""))
        return "\n".join(lines)


__all__ = [
    "EmissionFactorLookup",
    "EmissionFactorSearch",
    "EmissionsCalculator",
]
