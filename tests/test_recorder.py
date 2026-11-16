"""The second-recording script, run against a stand-in SDK and a stand-in clock. Authored-by: CPCS.

No robot, no network, no MuJoCo. hardware/record_barriered.py is loaded with
a fake anki_vector package and a clock that only moves when the script
sleeps, then its own main() is called. What is checked is the sequence it
would send and what it would write, against the "next recording" table in
the README. It says nothing about how a real robot answers.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import _common  # noqa: F401  (puts the skill on the path)
from simulacrum import recording

SCRIPT = Path(__file__).resolve().parent.parent / "hardware" / "record_barriered.py"
ENV_KEYS = ("PHASES", "RULER_PAUSE_S", "EDGE_SPEED", "EDGE_SECONDS", "EDGE_START_MM", "ABORT_WATCH_S", "NOTE")
TIME_TOL_S = 0.05
MOVING = ("wheels", "head", "lift", "drive_straight", "turn_in_place", "drive_off_charger")


class FakeClock:
    """Time that moves only when the script sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0, "negative sleep"
        self.now += seconds


class FakeRobot:
    """Records every command with the clock's time. Flags can be set to come up at a given time."""

    def __init__(self, clock: FakeClock, control: bool = True, on_charger: bool = False) -> None:
        self.clock, self.calls, self.on_charger = clock, [], on_charger
        self.cliff_after_first_wheels: Optional[float] = None
        self.picked_up_at: Optional[float] = None
        self.control_until: Optional[float] = None if control else -1.0
        self.stale_flag = False
        granted = types.SimpleNamespace(is_set=lambda: self.has_control)
        self.conn = types.SimpleNamespace(_control_events=types.SimpleNamespace(has_control=True),
                                          control_granted_event=granted)
        self.motors = types.SimpleNamespace(set_wheel_motors=self._wheels, stop_all_motors=lambda: self._call("stop"))
        self.behavior = types.SimpleNamespace(
            set_head_angle=lambda a: self._block("head", a), set_lift_height=lambda h: self._block("lift", h),
            drive_straight=lambda d, s, should_play_anim=True: self._block("drive_straight", d, s, should_play_anim),
            turn_in_place=lambda a, speed=None: self._block("turn_in_place", a, speed),
            drive_off_charger=lambda: self._block("drive_off_charger"))
        self.events = types.SimpleNamespace(subscribe=lambda *a: None, unsubscribe=lambda *a: None)
        self.status = self

    def connect(self, timeout: float = 0.0) -> None:
        self._call("connect")

    def disconnect(self) -> None:
        self._call("disconnect")

    def _call(self, name: str, *args: Any) -> None:
        self.calls.append((round(self.clock.now, 4), name) + args)

    def _wheels(self, left: float, right: float) -> None:
        self._call("wheels", left, right)

    def _block(self, name: str, *args: Any) -> Any:
        self._call(name, *args)
        self.clock.sleep(0.5)
        return types.SimpleNamespace(result="ok")

    @property
    def has_control(self) -> bool:
        return self.control_until is None or self.clock.now < self.control_until

    @property
    def is_cliff_detected(self) -> bool:
        first = [call[0] for call in self.calls if call[1] == "wheels"]
        return self.cliff_after_first_wheels is not None and bool(first) and (
            self.clock.now >= first[0] + self.cliff_after_first_wheels)

    @property
    def is_picked_up(self) -> bool:
        return self.picked_up_at is not None and self.clock.now >= self.picked_up_at

    is_being_held = is_falling = False

    @property
    def is_on_charger(self) -> bool:
        return self.on_charger and not any(call[1] == "drive_off_charger" for call in self.calls)


def _fake_sdk(robot: FakeRobot) -> Dict[str, Any]:
    """Stand-ins for the four anki_vector modules the script imports."""
    top = types.ModuleType("anki_vector")
    top.Robot = lambda *args, **kwargs: robot
    conn = types.ModuleType("anki_vector.connection")
    conn.ControlPriorityLevel = types.SimpleNamespace(DEFAULT_PRIORITY="default")
    events = types.ModuleType("anki_vector.events")
    events.Events = types.SimpleNamespace(robot_state="robot_state")
    util = types.ModuleType("anki_vector.util")
    util.degrees, util.distance_mm, util.speed_mmps = (lambda x: ("deg", x)), (lambda x: ("mm", x)), (lambda x: ("mmps", x))
    return {"anki_vector": top, "anki_vector.connection": conn, "anki_vector.events": events, "anki_vector.util": util}


def _load_script(robot: FakeRobot, lock: str) -> Any:
    with mock.patch.dict(sys.modules, _fake_sdk(robot)):
        spec = importlib.util.spec_from_file_location("record_barriered_under_test", SCRIPT)
        assert spec is not None and spec.loader is not None, "cannot load the recorder"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    module.time, module.T0, module.LOCK = robot.clock, 0.0, lock
    return module


def dry_run(env: Dict[str, str], robot: Optional[FakeRobot] = None) -> Dict[str, Any]:
    """Run the script's own main() against the stand-ins. Returns its rows, the robot's calls and the exit code."""
    robot = robot if robot is not None else FakeRobot(FakeClock())
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
        for key in ENV_KEYS:
            if key not in env:
                os.environ.pop(key, None)
        module = _load_script(robot, str(Path(tmp) / "lock"))
        try:
            with contextlib.redirect_stdout(out):
                code = module.main()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        lock_left = Path(module.LOCK).exists()
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    return {"rows": rows, "calls": robot.calls, "code": code, "lock_left": lock_left, "module": module}


def named(rows: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("name") == name]


def held_for(calls: List[tuple]) -> List[float]:
    """For each wheel command, the time until the next stop."""
    out = []
    for index, call in enumerate(calls):
        if call[1] == "wheels":
            stops = [later[0] for later in calls[index + 1:] if later[1] == "stop"]
            assert stops, "a wheel command was never stopped"
            out.append(round(stops[0] - call[0], 4))
    return out


class Sequence(unittest.TestCase):
    def test_default_phases_are_rest_then_head_and_lift_then_the_fast_spins(self) -> None:
        run = dry_run({})
        rows = run["rows"]
        self.assertEqual(run["code"], 0)
        self.assertEqual([row["phase"] for row in named(rows, "phase")], ["rest", "headlift", "spins"])
        self.assertAlmostEqual(named(rows, "phase")[1]["t"] - named(rows, "rest")[0]["t"], 20.0, delta=TIME_TOL_S)
        self.assertEqual([row["deg"] for row in named(rows, "head")], [-22.0, 45.0, -22.0, 45.0, -22.0, 10.0, -22.0, 0.0])
        self.assertEqual([row["norm"] for row in named(rows, "lift")], [0.0, 1.0, 0.0, 1.0, 0.0, 0.4667, 0.0])
        spins = [(row["label"], row["left"], row["right"]) for row in named(rows, "wheels")]
        self.assertEqual(spins, [("spin_ccw_150", -150.0, 150.0), ("spin_cw_150", 150.0, -150.0),
                                 ("spin_ccw_220", -220.0, 220.0), ("spin_cw_220", 220.0, -220.0)])
        for seconds in held_for(run["calls"]):
            self.assertAlmostEqual(seconds, 3.0, delta=TIME_TOL_S)

    def test_lift_part_move_is_60_mm(self) -> None:
        nominal = recording_nominal()
        span = nominal["lift_height_max_mm"] - nominal["lift_height_min_mm"]
        self.assertAlmostEqual(nominal["lift_height_min_mm"] + 0.4667 * span, 60.0, delta=0.01)

    def test_straights_hold_two_seconds_each_way_and_pause_for_the_ruler(self) -> None:
        run = dry_run({"PHASES": "straights", "RULER_PAUSE_S": "30"})
        wheels = named(run["rows"], "wheels")
        self.assertEqual([row["label"] for row in wheels], ["forward_50", "back_50", "forward_100", "back_100",
                                                            "forward_150", "back_150", "forward_200", "back_200"])
        self.assertEqual([row["left"] for row in wheels], [50.0, -50.0, 100.0, -100.0, 150.0, -150.0, 200.0, -200.0])
        self.assertTrue(all(row["left"] == row["right"] for row in wheels))
        for seconds in held_for(run["calls"]):
            self.assertAlmostEqual(seconds, 2.0, delta=TIME_TOL_S)
        self.assertEqual([row["after"] for row in named(run["rows"], "ruler_pause")],
                         ["forward_50", "forward_100", "forward_150", "forward_200"])
        gap = wheels[1]["t"] - wheels[0]["t"]
        self.assertAlmostEqual(gap, 2.0 + 2.0 + 30.0, delta=2 * TIME_TOL_S)

    def test_reversal_goes_straight_into_reverse_with_no_stop_between(self) -> None:
        run = dry_run({"PHASES": "reversal"})
        motion = [call for call in run["calls"] if call[1] in ("wheels", "stop")]
        self.assertEqual([call[1:] for call in motion[:3]], [("wheels", 100.0, 100.0), ("wheels", -100.0, -100.0), ("stop",)])
        self.assertAlmostEqual(motion[1][0] - motion[0][0], 1.0, delta=TIME_TOL_S)
        self.assertAlmostEqual(motion[2][0] - motion[1][0], 1.0, delta=TIME_TOL_S)
        self.assertEqual([row["label"] for row in named(run["rows"], "wheels")], ["reversal_100", "reversal_100_then"])

    def test_arcs_then_the_robots_own_actions(self) -> None:
        run = dry_run({"PHASES": "arcs,actions"}, FakeRobot(FakeClock(), on_charger=True))
        self.assertEqual(run["calls"][1][1], "drive_off_charger")
        arcs = [(row["left"], row["right"], row["seconds"]) for row in named(run["rows"], "wheels")]
        self.assertEqual(arcs, [(60.0, 120.0, 3.0), (120.0, 60.0, 3.0)])
        drives = [call[2:] for call in run["calls"] if call[1] == "drive_straight"]
        self.assertEqual(drives, [(("mm", 100.0), ("mmps", 50.0), False), (("mm", -100.0), ("mmps", 50.0), False),
                                  (("mm", 100.0), ("mmps", 100.0), False), (("mm", -100.0), ("mmps", 100.0), False)])
        turns = [call[2:] for call in run["calls"] if call[1] == "turn_in_place"]
        self.assertEqual(turns, [(("deg", 90.0), ("deg", 60.0)), (("deg", -90.0), ("deg", 60.0))])

    def test_repeat_is_the_spin_and_the_steps_of_the_first_recording(self) -> None:
        _, commands = recording.load()
        first = {row["label"]: (row["left"], row["right"], row["seconds"]) for row in commands if row["name"] == "wheels"}
        again = [(row["left"], row["right"], row["seconds"]) for row in named(dry_run({"PHASES": "repeat"})["rows"], "wheels")]
        self.assertEqual(again, [first["spin_ccw_60"], first["back_80"], first["forward_80"]])


class Safety(unittest.TestCase):
    def test_every_wheel_command_is_inside_the_limit_and_is_stopped(self) -> None:
        run = dry_run({"PHASES": "spins,straights,reversal,arcs,repeat"})
        speeds = [abs(value) for call in run["calls"] if call[1] == "wheels" for value in call[2:]]
        self.assertEqual(len(speeds), 2 * 19)
        self.assertLessEqual(max(speeds), 220.0)
        self.assertLessEqual(max(held_for(run["calls"])), 3.0 + TIME_TOL_S)
        self.assertEqual([call[1] for call in run["calls"][-2:]], ["stop", "disconnect"])
        self.assertFalse(run["lock_left"])

    def test_nothing_is_commanded_without_control(self) -> None:
        run = dry_run({"PHASES": "spins,actions"}, FakeRobot(FakeClock(), control=False))
        self.assertEqual(len(named(run["rows"], "no_control")), 1)
        self.assertEqual([call[1] for call in run["calls"] if call[1] in MOVING], [])

    def test_a_pick_up_ends_every_later_phase(self) -> None:
        robot = FakeRobot(FakeClock())
        robot.picked_up_at = 2.5
        run = dry_run({"PHASES": "spins,straights,arcs"}, robot)
        self.assertEqual(len([call for call in run["calls"] if call[1] == "wheels"]), 1)
        self.assertEqual([row["phase"] for row in named(run["rows"], "phase")], ["spins"])
        self.assertEqual(named(run["rows"], "end")[0]["abort"], "picked_up")
        self.assertEqual([call[1] for call in run["calls"] if call[0] > 2.6 and call[1] in MOVING], [])

    def test_losing_control_ends_the_run(self) -> None:
        robot = FakeRobot(FakeClock())
        robot.control_until = 3.0
        run = dry_run({"PHASES": "straights"}, robot)
        self.assertEqual(named(run["rows"], "end")[0]["abort"], "control_lost")
        self.assertEqual([call[1] for call in run["calls"] if call[0] > 3.1 and call[1] in MOVING], [])

    def test_control_is_read_from_the_granted_event_not_the_flag_that_stays_set(self) -> None:
        robot = FakeRobot(FakeClock())
        robot.control_until = 3.0
        self.assertTrue(robot.conn._control_events.has_control)
        run = dry_run({"PHASES": "spins"}, robot)
        self.assertTrue(robot.conn._control_events.has_control, "the stand-in flag stays set, as in the SDK")
        self.assertEqual(named(run["rows"], "end")[0]["abort"], "control_lost")
        self.assertEqual(len([call for call in run["calls"] if call[1] == "wheels"]), 1)

    def test_a_held_lock_means_no_connection(self) -> None:
        robot = FakeRobot(FakeClock())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PHASES": "rest"}):
            module = _load_script(robot, str(Path(tmp) / "lock"))
            Path(module.LOCK).write_text("held by another run\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 3)
        self.assertEqual(robot.calls, [])


class Edge(unittest.TestCase):
    ENV = {"PHASES": "edge", "EDGE_SPEED": "60", "EDGE_START_MM": "100", "NOTE": "white board, book barrier"}

    def test_edge_phase_refuses_before_connecting_without_the_ruler_distance(self) -> None:
        for start in (None, "0", "400"):
            robot = FakeRobot(FakeClock())
            env = {key: value for key, value in self.ENV.items() if key != "EDGE_START_MM"}
            env.update({} if start is None else {"EDGE_START_MM": start})
            with self.assertRaises(AssertionError, msg=str(start)):
                dry_run(env, robot)
            self.assertEqual(robot.calls, [], start)

    def test_the_ruler_distance_and_the_note_are_in_the_file(self) -> None:
        rows = dry_run(self.ENV)["rows"]
        self.assertEqual(named(rows, "edge_setup")[0]["start_to_edge_mm"], 100.0)
        self.assertEqual(named(rows, "begin")[0]["note"], "white board, book barrier")
        self.assertEqual((named(rows, "wheels")[0]["left"], named(rows, "wheels")[0]["seconds"]), (60.0, 3.0))

    def test_after_the_cliff_it_stops_then_only_listens_for_six_seconds(self) -> None:
        robot = FakeRobot(FakeClock())
        robot.cliff_after_first_wheels = 1.5
        run = dry_run(self.ENV, robot)
        wheels_at = [call[0] for call in run["calls"] if call[1] == "wheels"]
        stops = [row for row in named(run["rows"], "stop") if row["why"] == "cliff_detected"]
        self.assertEqual((len(wheels_at), len(stops)), (1, 1))
        self.assertAlmostEqual(stops[0]["t"] - wheels_at[0], 1.5, delta=TIME_TOL_S)
        self.assertEqual([row["seconds"] for row in named(run["rows"], "abort_watch")], [6.0])
        self.assertGreaterEqual(named(run["rows"], "end")[0]["t"] - stops[0]["t"], 6.0)
        after = [call[1] for call in run["calls"] if call[0] > stops[0]["t"] + TIME_TOL_S]
        self.assertEqual(after, ["stop", "disconnect"])


class Rows(unittest.TestCase):
    def test_state_rows_carry_what_the_replay_reads_and_the_image_time(self) -> None:
        module = dry_run({"PHASES": "rest"})["module"]
        vec = types.SimpleNamespace(x=1.0, y=2.0, z=3.0)
        message = types.SimpleNamespace(
            gyro=vec, accel=vec, pose=vec, left_wheel_speed_mmps=4.0, right_wheel_speed_mmps=5.0, head_angle_rad=0.1,
            lift_height_mm=32.0, pose_angle_rad=0.2, pose_pitch_rad=0.3, status=7, last_image_time_stamp=123456,
            prox_data=types.SimpleNamespace(distance_mm=200, signal_quality=0.5))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            module.on_state(None, None, message)
        row = json.loads(out.getvalue())
        with open(recording.RECORDING, encoding="utf-8") as handle:
            first = next(json.loads(line) for line in handle if json.loads(line)["k"] == "s")
        self.assertEqual(set(row), set(first) | {"img_ts"})
        self.assertEqual((row["wl"], row["wr"], row["gyro"], row["img_ts"]), (4.0, 5.0, [1.0, 2.0, 3.0], 123456))

    def test_plain_wheel_segments_load_with_the_first_recordings_reader(self) -> None:
        commands = [row for row in dry_run({"PHASES": "spins,straights"})["rows"] if row["k"] == "c"]
        segments = recording._wheel_segments(commands)
        self.assertEqual(len(segments), 12)
        self.assertEqual((segments[0].label, segments[0].left, segments[0].right), ("spin_ccw_150", -150.0, 150.0))
        for segment in segments:
            self.assertAlmostEqual(segment.t_stop - segment.t_cmd, 3.0 if "spin" in segment.label else 2.0, delta=TIME_TOL_S)


def recording_nominal() -> Dict[str, Any]:
    from simulacrum import params
    return params.values(params.load())


if __name__ == "__main__":
    unittest.main()
