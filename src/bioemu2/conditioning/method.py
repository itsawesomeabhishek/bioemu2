"""Method conditioning."""

from typing import Literal

SamplingMode = Literal["structure", "equilibrium"]


_METHOD_TYPES_IDS_RAW: dict[str, int] = {
    "X-RAY DIFFRACTION": 0,
    "ELECTRON MICROSCOPY": 1,
    "SOLUTION NMR": 2,
    "SOLID-STATE NMR": 3,
    "NEUTRON DIFFRACTION": 3,
    "ELECTRON CRYSTALLOGRAPHY": 3,
    "FIBER DIFFRACTION": 3,
    "POWDER DIFFRACTION": 3,
    "INFRARED SPECTROSCOPY": 3,
    "FLUORESCENCE TRANSFER": 3,
    "EPR": 3,
    "THEORETICAL MODEL": 3,
    "SOLUTION SCATTERING": 3,
    "OTHER": 3,
    "MD": 4,
    "AFDB": 5,
    "AFDB-OPENFOLD": 6,
    "NON-MD": 7,
    "FUTURE2": 8,
    "FUTURE3": 9,
    "FUTURE4": 10,
    "FUTURE5": 11,
    "FUTURE6": 12,
    "FUTURE7": 13,
    "FUTURE8": 14,
    "FUTURE9": 15,
    "FUTURE10": 16,
}
method_types_ids: dict[str, int] = {k.lower(): v for k, v in _METHOD_TYPES_IDS_RAW.items()}
num_method_types: int = len(set(method_types_ids.values()))

__all__ = [
    "SamplingMode",
    "method_types_ids",
    "num_method_types",
]
