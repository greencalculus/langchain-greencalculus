"""Sourced greenhouse-gas emission factors as LangChain tools.

Every value returns with the publisher's citation and the exact place it came
from, because an LLM recalling a factor from memory is right about 46% of the
time and gives the right publisher with the wrong number in up to 59% of
answers. Measured, not asserted: https://doi.org/10.5281/zenodo.22692277
"""

from langchain_greencalculus.tools import (
    EmissionFactorLookup,
    EmissionFactorSearch,
    EmissionsCalculator,
    ShipCIICalculator,
)

__version__ = "0.2.0"

__all__ = [
    "EmissionFactorLookup",
    "EmissionFactorSearch",
    "EmissionsCalculator",
    "ShipCIICalculator",
    "__version__",
]
