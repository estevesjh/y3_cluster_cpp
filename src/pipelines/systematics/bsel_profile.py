"""Shared BselModels dispatcher for tagged selection-bias profiles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .BselCostanzi26 import BselCostanzi26, SECTION as COSTANZI_SECTION
from .BselSunayama23 import BselSunayama23, SECTION as SUNAYAMA_SECTION


@dataclass(frozen=True)
class BselModels:
    """Dispatch one explicit Bsel tag to its validated kernel."""

    tag: str
    model: BselCostanzi26 | BselSunayama23

    def __post_init__(self):
        if self.tag not in ("Costanzi26", "Sunayama23"):
            raise ValueError("BselModels tag must be Costanzi26 or Sunayama23")
        if self.tag == "Costanzi26" and not isinstance(self.model, BselCostanzi26):
            raise TypeError("Costanzi26 requires a BselCostanzi26 model")
        if self.tag == "Sunayama23" and not isinstance(self.model, BselSunayama23):
            raise TypeError("Sunayama23 requires a BselSunayama23 model")

    @classmethod
    def from_source(cls, source: Any, tag: str, section: str | None = None):
        """Construct the tagged kernel from a CosmoSIS DataBlock section."""
        if tag == "Costanzi26":
            return cls(tag, BselCostanzi26.from_source(
                source, COSTANZI_SECTION if section is None else section))
        if tag == "Sunayama23":
            return cls(tag, BselSunayama23.from_source(
                source, SUNAYAMA_SECTION if section is None else section))
        raise ValueError("BselModels tag must be Costanzi26 or Sunayama23")

    def __call__(self, R, *, bin_index, lob=None, z=None):
        """Evaluate the selected Bsel model with an explicit wall context."""
        if self.tag == "Costanzi26":
            if lob is None or z is None:
                raise ValueError("Costanzi26 requires lob and z")
            return self.model(R, lob, z)
        return self.model(R, bin_index)

    def derivative(self, R, *, bin_index, lob=None, z=None):
        """Evaluate the selected model derivative."""
        if self.tag == "Costanzi26":
            if lob is None or z is None:
                raise ValueError("Costanzi26 requires lob and z")
            return self.model.derivative(R, lob, z)
        return self.model.derivative(R, bin_index)
