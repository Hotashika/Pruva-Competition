import ast
import types
from pathlib import Path


BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "bridge"
    / "bridge_node.py"
)


def _load_set_global_velocity():
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "set_global_velocity"
    )
    namespace = {
        "time": types.SimpleNamespace(time=lambda: 12.0),
        "mavutil": types.SimpleNamespace(
            mavlink=types.SimpleNamespace(
                MAV_FRAME_GLOBAL_RELATIVE_ALT=3,
                MAVLink_set_position_target_global_int_message=(
                    lambda *args: args
                ),
            )
        ),
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(BRIDGE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["set_global_velocity"]


class _Mav:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def test_sends_velocity_only_global_target_in_metres_per_second():
    mav = _Mav()
    connection = types.SimpleNamespace(
        target_system=1,
        target_component=2,
        mav=mav,
    )

    _load_set_global_velocity()(
        connection,
        north_m_s=0.8,
        east_m_s=-0.6,
        boot_time=10.0,
    )

    assert len(mav.messages) == 1
    message = mav.messages[0]
    assert message[0] == 2000
    assert message[1:4] == (1, 2, 3)
    assert message[4] == 0b110111100111
    assert message[5:8] == (0, 0, 0)
    assert message[8:11] == (0.8, -0.6, 0)
