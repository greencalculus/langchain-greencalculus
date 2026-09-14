# langchain-greencalculus

LangChain tools for **sourced** greenhouse-gas emission factors. Every value
comes back with the publisher, the exact place it was read from, and a citation
your agent can repeat.

[![PyPI](https://img.shields.io/pypi/v/langchain-greencalculus)](https://pypi.org/project/langchain-greencalculus/)
[![MIT](https://img.shields.io/badge/licence-MIT-blue)](./LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22692277.svg)](https://doi.org/10.5281/zenodo.22692277)
[![Listed in LangChain docs](https://img.shields.io/badge/LangChain-integrations%20directory-1C3C3C)](https://docs.langchain.com/oss/python/integrations/tools)

```bash
pip install langchain-greencalculus
```

## Why this exists

We put 467 emission-factor questions to five frontier models with no tools.
They were correct **46%** of the time — and, worse, they named the *right
publisher* while giving the *wrong number* in up to **59%** of answers.

That is the failure that survives review. A wrong number with no source gets
caught. A wrong number wearing DEFRA's name does not.

Given a sourced lookup tool, the same models reached **99–100%**.

Those figures are measured and reproducible, not marketing: the questions, every
raw model answer and the scoring code are published, with a DOI
([10.5281/zenodo.22692277](https://doi.org/10.5281/zenodo.22692277)) and on
[HuggingFace](https://huggingface.co/datasets/greencalculus/emission-factor-benchmark).
The study also reports the parts that reflect badly on us.

**This package is the fix that study describes.**

## No API key needed to start

The corpus is open to read, so the tools work before anyone signs up:

```python
from langchain_greencalculus import EmissionFactorLookup

print(EmissionFactorLookup().invoke({"key": "grid.gbr.electricity.location_based"}))
```

```
0.13096 kg CO2e per kWh
key: grid.gbr.electricity.location_based
citation: UK grid electricity — location-based (generation). UK Government GHG
Conversion Factors 2026 — Department for Energy Security and Net Zero (DESNZ),
cell 'UK electricity'!E25, retrieved 2026-06-18. via GreenCalculus data version
2026.189, factor grid.gbr.electricity.location_based.
https://verify.greencalculus.com/grid.gbr.electricity.location_based@2026.189
```

That cell reference is the point. The number is traceable to a row in a
government workbook, not to a blog post.

## Use it in an agent

```python
from langchain.agents import create_agent
from langchain_greencalculus import (
    EmissionFactorLookup, EmissionFactorSearch, EmissionsCalculator,
)

tools = [EmissionFactorSearch(), EmissionFactorLookup(), EmissionsCalculator()]
agent = create_agent("anthropic:claude-sonnet-4-5", tools=tools)

agent.invoke({"messages": [{"role": "user",
    "content": "We used 12,000 kWh of UK grid electricity. What are the emissions, and cite the source."}]})
```

The tool descriptions tell the model to search before guessing a key, to prefer
the tool over recall, and to repeat the citation. They also warn it about the
distinctions that silently produce a plausible wrong answer — location-based vs
market-based, well-to-tank vs combustion.

## The tools

| Tool | Needs a key | What it does |
|---|---|---|
| `EmissionFactorSearch` | no | Plain-English search, returns candidate keys with values and publishers |
| `EmissionFactorLookup` | no | One factor by exact key, with its citation. `as_of` pinning needs a key |
| `EmissionsCalculator` | optional | Quantity × factor. With a key you get a server-side audit trail; without one it multiplies locally **and says so** |

## Configuration

```python
EmissionFactorLookup(api_key="gc_live_…")      # explicit
```

Or set `GREENCALCULUS_API_KEY` and construct with no arguments. A free key
(1,000 calls/month, no card) is at
[greencalculus.com/developers](https://greencalculus.com/developers/?ref=langchain).

A key adds: `as_of` version pinning so a figure reproduces a year later, traced
server-side calculations, and higher rate limits.

## Standard tests

This package runs LangChain's own conformance suite — `ToolsUnitTests` and
`ToolsIntegrationTests` from
[`langchain-tests`](https://pypi.org/project/langchain-tests/) — against all
three tools, sync and async.

```bash
pip install -e ".[test]" && pytest
```

The integration tests hit the live open route, so they pass with **no secret
configured**. Nothing here is mocked into looking like it works.

Listed in LangChain's own integration docs — as
[`GreenCalculusToolkit`](https://docs.langchain.com/oss/python/integrations/tools)
in the Python tools directory, and as a provider in
[all providers](https://docs.langchain.com/oss/python/integrations/providers/all_providers).

## Licence

MIT. The emission factors themselves carry their publishers' licences, which are
returned with each value and
[audited in full](https://greencalculus.com/guides/emission-factor-licences/) —
42% of the 137 sources we read carry no standard licence at all.
