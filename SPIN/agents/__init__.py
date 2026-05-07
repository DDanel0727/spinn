"""
SPIN two-stage participant simulation (personality compilation, then in-trial elicitation and decision).

Implementation lives under ``agents`` (the former nested ``spin`` package was flattened).
"""

from .runner import SpinParticipantAgent

__all__ = ["SpinParticipantAgent"]
