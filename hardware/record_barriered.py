"""Second recording of Vector for the simulacrum: the sequence a BARRIERED surface allows.

Authored-by: CPCS (cpcs-5f62f24). Runs on the wire-pod host with its anki_vector install,
fed on stdin; writes JSON lines to stdout and leaves nothing behind but a lock file it removes.

NOT RUN YET. This copy is kept in skills/vector-simulacrum so the sequence the model needs is
written down beside the model. Nothing in the skill imports or runs it against a robot;
tests/test_recorder.py runs it against a stand-in SDK with a stand-in clock.
Edited by cpcs-66ffc23, logging only, from the original with sha256
192269145c74c5eb3e5f523a232d845cac9ef769702426a1dc770afcefad8602:
after a hazard it keeps recording for ABORT_WATCH_S without commanding anything, the edge
phase needs and records EDGE_START_MM, and NOTE is copied into the begin row.

DO NOT RUN ON AN OPEN-EDGED TABLE. The straight runs need up to 450 mm of clear travel
(phase "straights" at 200 mm/s for 2.0 s, plus stopping), and the fast spins and arcs can
walk the robot sideways. The surface needs a wall, or a soft landing, on every side.

Choose phases with the environment variable PHASES (comma separated, default "rest,headlift,spins"):
  rest       stand still for 20 s, awake                              no travel
  headlift   head and lift over their whole range, and part moves     no travel
  spins      spins for 3.0 s both ways at 150 and 220 mm/s            turns on the spot; allow 100 mm all round
  straights  forward then back for 2.0 s at 50, 100, 150, 200 mm/s    up to 450 mm forward of the start
  reversal   forward 100 mm/s for 1.0 s straight into back 1.0 s      about 120 mm forward
  arcs       left 60 / right 120 mm/s for 3.0 s, then mirrored        a circle of about 150 mm radius
  actions    drive_straight 100 mm at 50 and 100 mm/s; turn_in_place  about 120 mm forward
             90 degrees at 60 deg/s both ways
  repeat     one 60 mm/s spin and one 80 mm/s step, as in run 1       about 30 mm; put it LAST in PHASES
  edge       ONE approach to an edge at EDGE_SPEED mm/s (default 30)  drives until the robot's own cliff
             for at most EDGE_SECONDS (default 3.0)                    reaction stops it

The edge phase. Before the run, with the robot standing square to the edge, measure with a ruler
from the line where the front wheels touch the surface to the edge, and pass it as EDGE_START_MM.
The phase refuses to start without it. Nothing can be measured by hand afterwards: about 1.2 s
after it stops the robot backs away by itself and may turn. The stop point is the start distance
less the travel the robot reports. Start between EDGE_SPEED x 1.0 s and EDGE_SPEED x
(EDGE_SECONDS - 0.5 s) from the edge, so the robot is at speed when it arrives and arrives in time.

ABORT_WATCH_S (default 6): after a cliff or any other hazard ends the run, the script sends its
stop and then keeps recording for this long without commanding anything, so the robot's own
reaction (the pause, the move away, the turn) is in the file.

NOTE: free text copied into the begin row (the surface, the barrier, ruler readings).

img_ts is the robot's time stamp of its last camera image. It shows how evenly state messages
arrive. It cannot split the delay of a command going in from the delay of state coming back:
the two clocks share no event.

RULER_PAUSE_S (default 0): seconds to stand still after each forward leg of "straights", so the
distance can be marked and measured before the robot drives back.

Safety:
- Control is requested at DEFAULT priority, so the robot's own cliff reaction stays on and
  can take control away. If control is not granted at the start (it is refused while the
  cliff flag is set), the script exits without commanding anything.
- Every motion is bounded in time and followed by stop_all_motors.
- A cliff, pick-up, fall or held flag, or loss of control, ends all motion for the run.
  The script then only listens (ABORT_WATCH_S) and commands nothing more.
- A blocking behaviour call (head, lift, drive_straight, turn_in_place, drive_off_charger) has
  no time limit of its own here. It is not started without control, but if control is lost
  while it runs it may not return: end the script with SIGTERM, which stops the motors.
- It does not try to dock. The robot is left where the sequence ends.
"""
import json
import os
import signal
import sys
import threading
import time

import anki_vector
from anki_vector.connection import ControlPriorityLevel
from anki_vector.events import Events
from anki_vector.util import degrees, distance_mm, speed_mmps

SERIAL = os.environ.get("VECTOR_SERIAL", "00000000")  # set VECTOR_SERIAL to your robot's serial number
LOCK = "/tmp/vector-%s-takeover.lock" % SERIAL
POLL_S = 0.02
MAX_WHEEL_MMPS = 220.0
MAX_SEGMENT_S = 3.5
MAX_WATCH_S = 30.0
KNOWN_PHASES = ("rest", "headlift", "spins", "straights", "reversal", "arcs", "actions", "repeat", "edge")
OUT_LOCK = threading.Lock()
T0 = time.monotonic()
STATE = {"robot": None, "abort": None}


def emit(kind, **fields):
    """One JSON line, safe from any thread."""
    row = {"k": kind, "t": round(time.monotonic() - T0, 4)}
    row.update(fields)
    with OUT_LOCK:
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()


def on_state(robot, event_type, m):
    emit("s", gyro=[m.gyro.x, m.gyro.y, m.gyro.z], accel=[m.accel.x, m.accel.y, m.accel.z],
         wl=m.left_wheel_speed_mmps, wr=m.right_wheel_speed_mmps, head=m.head_angle_rad, lift=m.lift_height_mm,
         pose=[m.pose.x, m.pose.y, m.pose.z], yaw=m.pose_angle_rad, pitch=m.pose_pitch_rad, status=m.status,
         prox=m.prox_data.distance_mm, proxq=m.prox_data.signal_quality, img_ts=m.last_image_time_stamp)


def has_control(robot):
    """Control as the connection's control_granted_event says. In wirepod_vector_sdk 0.8.1 the
    has_control flag is not cleared when control is lost, so it alone would miss a cliff reaction."""
    granted = getattr(robot.conn, "control_granted_event", None)
    if granted is not None:
        return bool(granted.is_set())
    events = getattr(robot.conn, "_control_events", None)
    return bool(getattr(events, "has_control", False))


def hazard(robot):
    """A reason to stop all motion, or None."""
    st = robot.status
    if st.is_cliff_detected:
        return "cliff_detected"
    if st.is_picked_up or st.is_being_held:
        return "picked_up"
    if st.is_falling:
        return "falling"
    if not has_control(robot):
        return "control_lost"
    return None


def stop(robot, why):
    try:
        robot.motors.stop_all_motors()
    except Exception as exc:  # best effort; the reason is still recorded
        emit("c", name="stop_failed", why=why, error=type(exc).__name__)
        return
    emit("c", name="stop", why=why)


def clear_to_move(robot, label):
    """False once the run has been aborted or a hazard shows; records why."""
    if STATE["abort"]:
        return False
    reason = hazard(robot)
    if reason:
        STATE["abort"] = reason
        emit("c", name="skip", label=label, why=reason)
        return False
    return True


def hold(robot, seconds):
    """Keep the present wheel command for a bounded time. False if a hazard ended it."""
    assert 0.0 < seconds <= MAX_SEGMENT_S, "segment too long"
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        reason = hazard(robot)
        if reason:
            STATE["abort"] = reason
            stop(robot, reason)
            return False
        time.sleep(POLL_S)
    return True


def wheels_for(robot, left, right, seconds, label, settle_s=2.0, then=None):
    """Two wheel speeds for a bounded time, optionally straight into a second pair, then stop and stand."""
    assert abs(left) <= MAX_WHEEL_MMPS and abs(right) <= MAX_WHEEL_MMPS, "wheel speed above the robot's limit"
    if not clear_to_move(robot, label):
        return False
    emit("c", name="wheels", label=label, left=left, right=right, seconds=seconds)
    robot.motors.set_wheel_motors(left, right)
    emit("c", name="wheels_sent", label=label)
    if not hold(robot, seconds):
        return False
    if then is not None:
        left2, right2, seconds2 = then
        assert abs(left2) <= MAX_WHEEL_MMPS and abs(right2) <= MAX_WHEEL_MMPS
        emit("c", name="wheels", label=label + "_then", left=left2, right=right2, seconds=seconds2)
        robot.motors.set_wheel_motors(left2, right2)
        emit("c", name="wheels_sent", label=label + "_then")
        if not hold(robot, seconds2):
            return False
    stop(robot, "segment_end")
    time.sleep(settle_s)
    return True


def behaviour(robot, name, call, **args):
    """One blocking SDK behaviour call, timed. Not started without control: it would block."""
    if not clear_to_move(robot, name):
        return False
    emit("c", name=name, **args)
    try:
        result = call()
        emit("c", name=name + "_done", result=str(getattr(result, "result", result))[:120].replace("\n", " "))
    except Exception as exc:
        emit("c", name=name + "_failed", error=("%s: %s" % (type(exc).__name__, exc))[:160])
        return False
    time.sleep(2.0)
    return True


def phase_rest(robot):
    emit("c", name="rest", seconds=20.0)
    time.sleep(20.0)


def phase_headlift(robot):
    b = robot.behavior
    for angle in (-22.0, 45.0, -22.0, 45.0, -22.0, 10.0, -22.0, 0.0):
        behaviour(robot, "head", lambda a=angle: b.set_head_angle(degrees(a)), deg=angle)
    for height in (0.0, 1.0, 0.0, 1.0, 0.0, 0.4667, 0.0):  # 0.4667 of the 32..92 mm range is 60 mm
        behaviour(robot, "lift", lambda h=height: b.set_lift_height(h), norm=height)


def phase_spins(robot):
    for speed in (150.0, 220.0):
        if not wheels_for(robot, -speed, speed, 3.0, "spin_ccw_%d" % speed):
            return
        if not wheels_for(robot, speed, -speed, 3.0, "spin_cw_%d" % speed):
            return


def phase_straights(robot):
    pause = float(os.environ.get("RULER_PAUSE_S", "0") or 0.0)
    assert 0.0 <= pause <= 120.0, "RULER_PAUSE_S out of range"
    for speed in (50.0, 100.0, 150.0, 200.0):
        if not wheels_for(robot, speed, speed, 2.0, "forward_%d" % speed):
            return
        if pause > 0.0:
            emit("c", name="ruler_pause", seconds=pause, after="forward_%d" % speed)
            time.sleep(pause)
        if not wheels_for(robot, -speed, -speed, 2.0, "back_%d" % speed):
            return


def phase_reversal(robot):
    wheels_for(robot, 100.0, 100.0, 1.0, "reversal_100", then=(-100.0, -100.0, 1.0))


def phase_arcs(robot):
    if wheels_for(robot, 60.0, 120.0, 3.0, "arc_left_60_120"):
        wheels_for(robot, 120.0, 60.0, 3.0, "arc_right_120_60")


def phase_actions(robot):
    b = robot.behavior
    for speed in (50.0, 100.0):
        behaviour(robot, "drive_straight", lambda s=speed: b.drive_straight(distance_mm(100.0), speed_mmps(s), should_play_anim=False), mm=100.0, mmps=speed)
        behaviour(robot, "drive_straight", lambda s=speed: b.drive_straight(distance_mm(-100.0), speed_mmps(s), should_play_anim=False), mm=-100.0, mmps=speed)
    for angle in (90.0, -90.0):
        behaviour(robot, "turn_in_place", lambda a=angle: b.turn_in_place(degrees(a), speed=degrees(60.0)), deg=angle, dps=60.0)


def phase_repeat(robot):
    if wheels_for(robot, -60.0, 60.0, 2.5, "repeat_spin_ccw_60"):
        if wheels_for(robot, -80.0, -80.0, 0.3125, "repeat_back_80"):
            wheels_for(robot, 80.0, 80.0, 0.3125, "repeat_forward_80")


def edge_settings():
    """Speed, time limit and the ruler-measured start distance of the edge phase. Raises if any is missing or out of range."""
    speed = float(os.environ.get("EDGE_SPEED", "30") or 30.0)
    seconds = float(os.environ.get("EDGE_SECONDS", "3.0") or 3.0)
    start = float(os.environ.get("EDGE_START_MM", "0") or 0.0)
    assert 10.0 <= speed <= 140.0, "EDGE_SPEED must be 10..140 mm/s"
    assert 0.5 <= seconds <= MAX_SEGMENT_S, "EDGE_SECONDS out of range"
    assert 5.0 <= start <= speed * seconds, "EDGE_START_MM (front wheel contact line to the edge, by ruler) is missing or out of reach"
    return speed, seconds, start


def phase_edge(robot):
    """One approach. The robot's own reaction is what stops it; this only bounds the time."""
    speed, seconds, start = edge_settings()
    emit("c", name="edge_setup", speed=speed, seconds=seconds, start_to_edge_mm=start)
    wheels_for(robot, speed, speed, seconds, "edge_approach_%d" % speed, settle_s=4.0)


def watch_after_abort():
    """After a hazard: command nothing, keep recording, so the robot's own reaction is in the file."""
    seconds = float(os.environ.get("ABORT_WATCH_S", "6") or 0.0)
    assert 0.0 <= seconds <= MAX_WATCH_S, "ABORT_WATCH_S out of range"
    emit("c", name="abort_watch", seconds=seconds, why=STATE["abort"])
    time.sleep(seconds)


PHASES = {"rest": phase_rest, "headlift": phase_headlift, "spins": phase_spins, "straights": phase_straights,
          "reversal": phase_reversal, "arcs": phase_arcs, "actions": phase_actions, "repeat": phase_repeat, "edge": phase_edge}


def chosen_phases():
    names = [n.strip() for n in os.environ.get("PHASES", "rest,headlift,spins").split(",") if n.strip()]
    unknown = [n for n in names if n not in KNOWN_PHASES]
    assert not unknown, "unknown phases: %s" % unknown
    assert names, "no phases chosen"
    if "edge" in names:
        edge_settings()  # refuse before connecting, not half way through the run
    return names


def sequence(robot, names):
    st = robot.status
    emit("c", name="begin", phases=names, control=has_control(robot), on_charger=bool(st.is_on_charger),
         cliff=bool(st.is_cliff_detected), note=os.environ.get("NOTE", "")[:400])
    if not has_control(robot):
        emit("c", name="no_control", why="not granted at DEFAULT priority; nothing was commanded")
        return
    if st.is_on_charger:
        if not behaviour(robot, "drive_off_charger", robot.behavior.drive_off_charger):
            return
    for name in names:
        if STATE["abort"]:
            break
        emit("c", name="phase", phase=name)
        PHASES[name](robot)
    if STATE["abort"]:
        watch_after_abort()
    emit("c", name="end", abort=STATE["abort"], control=has_control(robot), on_charger=bool(robot.status.is_on_charger))


def on_signal(signum, frame):
    robot = STATE["robot"]
    STATE["abort"] = "signal_%d" % signum
    if robot is not None:
        stop(robot, "signal")
    raise SystemExit(4)


def main():
    names = chosen_phases()
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        emit("c", name="lock_held", path=LOCK)
        return 3
    os.write(fd, ("cpcs-5f62f24 hardware recording pid %d\n" % os.getpid()).encode())
    os.close(fd)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    robot = anki_vector.Robot(SERIAL, behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY, cache_animation_lists=False)
    try:
        robot.connect(timeout=20)
        STATE["robot"] = robot
        robot.events.subscribe(on_state, Events.robot_state)
        try:
            time.sleep(1.0)
            sequence(robot, names)
        finally:
            stop(robot, "final")
            robot.events.unsubscribe(on_state, Events.robot_state)
            robot.disconnect()
    finally:
        os.remove(LOCK)
        emit("c", name="lock_released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
