import ast
import math
import types
from pathlib import Path


BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "bridge"
    / "bridge_node.py"
)


def _load_cmd_vel_callback():
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    bridge_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OrangeCubeBridgeNode"
    )
    method = next(
        node
        for node in bridge_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_cmd_vel_callback"
    )
    namespace = {
        "math": math,
        "time": types.SimpleNamespace(time=lambda: 12.0),
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(BRIDGE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["_cmd_vel_callback"]


def test_cmd_vel_rejects_non_finite_values_and_neutralizes_outputs():
    callback = _load_cmd_vel_callback()
    invalid_commands = (
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, float("-inf")),
    )

    for linear_x, angular_z in invalid_commands:
        errors = []
        node = types.SimpleNamespace(
            cmd_vel_ignored_reported=False,
            last_position_target_time=99.0,
            neutralize_count=0,
        )
        node._has_valid_link = lambda: True
        node._vehicle_ready_for_guided_motion = lambda _topic: True

        def neutralize_outputs():
            node.neutralize_count += 1
            node.last_position_target_time = 0.0

        node._neutralize_outputs = neutralize_outputs
        node._publish_error = errors.append
        msg = types.SimpleNamespace(
            linear=types.SimpleNamespace(x=linear_x),
            angular=types.SimpleNamespace(z=angular_z),
        )

        callback(node, msg)

        assert node.neutralize_count == 1
        assert node.last_position_target_time == 0.0
        assert errors == [
            "/cube/cmd_vel command contains a non-finite value; "
            "outputs were neutralized."
        ]
