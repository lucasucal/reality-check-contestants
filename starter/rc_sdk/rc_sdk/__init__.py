"""rc_sdk — the reality-check policy SDK (contract v0, stdlib only).

Implements the HTTP container contract from docs/platform.md §6.1 so teams
write a tiny Policy subclass instead of raw HTTP handling.

Contract rules encoded here:
- GET /healthz -> 200 {} immediately on start (LIVENESS ONLY, before weights load).
- GET /readyz  -> 200 {"ready": bool, "contract": "v0", "schema_version": 3,
                       "capabilities": {"chunking": true, "caching": false}}
                  ready flips true after Policy.load() finishes (background thread).
- POST /act    -> request envelope {schema_version, episode_id, step, task_id,
                  instruction, images, goal, proprio}. images and goal both map
                  wrist, left and right to 640×360 base64 RGB JPEGs.
                  response echoes episode_id and step VERBATIM plus "action":
                  {"episode_id": ..., "step": ..., "action": <Policy.act result>}.
                  Invalid requests (including incompatible versions) -> 400.
                  episode_id/step also echoed as headers X-RC-Episode-Id / X-RC-Step.

PURE-FUNCTION REQUIREMENT (/act): the same request envelope for the same
(episode_id, step) must yield a valid action — deterministic recompute suffices.
Keep external side effects out of act() and allow requests to arrive at any rate.

OBSERVATION/ACTION SCHEMA v3: 6 arm joint_delta values at up to ±0.05 rad
per step plus gripper. A class policy declares schema_version 3 in its manifest.
"""

from .policy import Policy
from .server import PolicyServer, main

__all__ = ["Policy", "PolicyServer", "main"]
__version__ = "0.1.0"
