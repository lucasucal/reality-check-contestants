"""Policy base class — subclass this and swap in your model.

Schema v3: 6 arm `joint_delta` values (|Δq| ≤ 0.05 rad per 50 ms step)
plus an absolute gripper target in [0, 1].
"""

from typing import Any, Dict


class Policy:
    """Your model behind the reality-check container contract (v0).

    Lifecycle:
      - `load()` runs in a background thread at server start; /readyz flips to
        ready when it returns. Load weights, build your model, warm up here.
      - `act(obs)` is called per step once ready.

    act() must be a PURE FUNCTION of the request envelope: same envelope for
    the same (episode_id, step) => valid action (deterministic recompute is
    fine). Keep external side effects out of act() and allow requests to arrive
    at any rate.
    """

    def load(self) -> None:
        """Load weights / initialize the model. Runs once, in the background.

        The judged runner mounts the selected public Hugging Face snapshot at
        /weights. The dashboard model link selects the current snapshot for each new submission.
        Load the mounted files rather than fetching a model by name at runtime.
        Do NOT commit weights to this repo.
        """
        raise NotImplementedError("implement load() to fetch/prepare your weights")

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """One observation in, one action dict out (schema v3: 6 joint_delta + gripper).

        The `obs` envelope contains schema_version, episode_id, step, task_id,
        instruction, images, goal and proprio. Both images and goal contain
        wrist, left and right 640×360 base64 RGB JPEGs. A policy can ignore
        goal if it relies on the instruction or other inputs. Return the
        action dict alone. The server echoes episode_id and step.
        """
        raise NotImplementedError("implement act() to run your model")
