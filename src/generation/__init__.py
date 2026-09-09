"""
Grounded Response Generation module with evidence traceability.
"""
from .prompts import SYSTEM_GROUNDING_PROMPT, build_grounding_prompt
from .generator import GroundedResponseGenerator, GroundedResponse

__all__ = [
    "SYSTEM_GROUNDING_PROMPT",
    "build_grounding_prompt",
    "GroundedResponseGenerator",
    "GroundedResponse"
]
