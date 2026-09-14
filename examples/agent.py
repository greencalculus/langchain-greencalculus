"""A working agent, end to end.

Run with no API key at all:   python examples/agent.py
The emission-factor tools need no signup; only the model provider does.
"""
import os

from langchain_greencalculus import (
    EmissionFactorLookup,
    EmissionFactorSearch,
    EmissionsCalculator,
)

TOOLS = [EmissionFactorSearch(), EmissionFactorLookup(), EmissionsCalculator()]

QUESTION = (
    "We used 12,000 kWh of UK grid electricity last month. "
    "What were the emissions? Cite the source for the factor you used."
)


def main() -> None:
    # The agent path calls a paid model provider. Opt in explicitly, so nobody
    # runs this file out of curiosity and gets a bill.
    if os.environ.get("RUN_AGENT") != "1":
        print("Set RUN_AGENT=1 (and ANTHROPIC_API_KEY) to run the full agent — "
              "that calls a paid model.\nShowing the tools directly instead, "
              "which costs nothing.\n")
        print(EmissionFactorSearch().invoke({"query": "UK grid electricity", "limit": 3}))
        print()
        print(EmissionsCalculator().invoke(
            {"key": "grid.gbr.electricity.location_based", "quantity": 12000}))
        return

    from langchain.agents import create_agent

    agent = create_agent("anthropic:claude-sonnet-5", tools=TOOLS)
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
