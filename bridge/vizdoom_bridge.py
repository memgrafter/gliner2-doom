#!/usr/bin/env python
"""ViZDoom engine glue for gliner2-doom. No app logic lives here.

JSON lines in on stdin, JSON lines out on stdout, logs on stderr. The Rust
binary is the clock and the policy; this process only owns the engine.

Commands (one JSON object per line):
  {"cmd":"init","scenario":"defend_the_center","mode":"sync","width":160,"height":120,
   "visible":false,"seed":null,"wad":null,"map":null,"buttons":null,"timeout_tics":null}
  {"cmd":"reset"}                                    -> observation
  {"cmd":"act","buttons":["TURN_LEFT"],"tics":4}      -> observation after the step
  {"cmd":"obs","frame":false}                        -> observation
  {"cmd":"close"}

Observation: {"ok":true,"tic":N,"finished":bool,"reward":float,"total_reward":float,
  "health":int,"ammo":int,"kills":int,"pos":[x,y,angle_deg],
  "actors":[{"name":str,"x":int,"y":int,"w":int,"h":int,"dist":float,"rel_angle":float}],
  "depth":{"left":float,"center":float,"right":float},   # mean depth of the middle band, thirds
  "frame":"<base64 RGB24 w*h*3>" (only when requested)}

`rel_angle` is degrees from the player's facing to the actor (+ = left, - = right),
from the labels' world positions; `dist` is world units. Both come straight from
the engine and are exact, unlike the screen-space x/w heuristics.
"""
import base64
import json
import math
import os
import sys

import numpy as np
import vizdoom as vzd

LEVEL_BUTTONS = ["ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT", "USE"]
GAME_VARS = [vzd.GameVariable.HEALTH, vzd.GameVariable.SELECTED_WEAPON_AMMO, vzd.GameVariable.KILLCOUNT,
             vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE, vzd.GameVariable.ITEMCOUNT]


def log(*a):
    print("[bridge]", *a, file=sys.stderr, flush=True)


class Bridge:
    def __init__(self):
        self.game = None
        self.buttons = []
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_vars = {}

    def init(self, scenario="defend_the_center", mode="sync", width=160, height=120, visible=False,
             seed=None, wad=None, map=None, buttons=None, timeout_tics=None, ticrate=None):
        g = vzd.DoomGame()
        if wad or scenario == "level":
            g.set_doom_game_path(wad or os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))
            g.set_doom_map(map or "map01")
            g.set_available_buttons([getattr(vzd.Button, b) for b in (buttons or LEVEL_BUTTONS)])
            g.set_episode_timeout(timeout_tics or 35 * 120)
            g.set_living_reward(0)
        else:
            g.load_config(os.path.join(vzd.scenarios_path, f"{scenario}.cfg"))
            if buttons:
                g.set_available_buttons([getattr(vzd.Button, b) for b in buttons])
            if timeout_tics:
                g.set_episode_timeout(timeout_tics)
        g.set_available_game_variables(GAME_VARS)
        g.set_window_visible(bool(visible))
        g.set_mode(vzd.Mode.ASYNC_PLAYER if mode == "async" else vzd.Mode.PLAYER)
        if ticrate:
            g.set_ticrate(int(ticrate))
        g.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{width}X{height}"))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_labels_buffer_enabled(True)
        g.set_depth_buffer_enabled(True)
        g.set_objects_info_enabled(False)
        g.set_render_hud(False)
        g.set_render_crosshair(True)
        g.set_render_all_frames(mode == "async")
        if seed is not None:
            g.set_seed(int(seed))
        g.init()
        self.game = g
        self.buttons = [b.name for b in g.get_available_buttons()]
        return {"ok": True, "buttons": self.buttons, "mode": mode}

    def reset(self):
        self.game.new_episode()
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_vars = {}
        return self.obs()

    def act(self, buttons=(), tics=1):
        vec = [1 if b in buttons else 0 for b in self.buttons]
        r = self.game.make_action(vec, int(tics))
        self.last_reward = float(r)
        self.total_reward += float(r)
        return self.obs()

    def obs(self, frame=False):
        g = self.game
        finished = g.is_episode_finished()
        out = {"ok": True, "finished": finished, "reward": self.last_reward, "total_reward": self.total_reward,
               "tic": g.get_episode_time()}
        s = None if finished else g.get_state()
        if s is None:
            # Terminal frame: the engine has no state, so report the last values seen
            # (otherwise kills/items/ammo read as zero at the end of every episode).
            out["finished"] = True
            out.update(self.last_vars)
            return out
        hp, ammo, kills, px, py, ang, items = (float(v) for v in s.game_variables)
        out.update(health=int(hp), ammo=max(int(ammo), 0), kills=int(kills), items=int(items), pos=[px, py, ang])
        self.last_vars = {"health": int(hp), "ammo": max(int(ammo), 0), "kills": int(kills), "items": int(items), "pos": [px, py, ang]}
        actors = []
        for l in s.labels:
            if l.object_name == "DoomPlayer":
                continue
            dx, dy = l.object_position_x - px, l.object_position_y - py
            dist = math.hypot(dx, dy)
            rel = (math.degrees(math.atan2(dy, dx)) - ang + 180.0) % 360.0 - 180.0
            actors.append({"name": l.object_name, "x": int(l.x), "y": int(l.y), "w": int(l.width), "h": int(l.height),
                           "dist": round(dist, 1), "rel_angle": round(rel, 1)})
        actors.sort(key=lambda a: a["dist"])
        out["actors"] = actors
        d = s.depth_buffer
        h, w = d.shape
        band = d[h // 3: 2 * h // 3, :].astype(np.float32)
        lower = d[h // 3: 5 * h // 6, :].astype(np.float32)  # middle + lower bands: catches steps and low obstacles
        third = slice(w // 3, 2 * w // 3)
        out["depth"] = {"left": float(band[:, : w // 3].mean()), "center": float(band[:, third].mean()),
                        "right": float(band[:, 2 * w // 3:].mean()),
                        "near": float(np.percentile(lower[:, third], 10)),
                        "near_left": float(np.percentile(lower[:, : w // 3], 10)),
                        "near_right": float(np.percentile(lower[:, 2 * w // 3:], 10))}
        if frame:
            out["frame"] = base64.b64encode(np.ascontiguousarray(s.screen_buffer).tobytes()).decode("ascii")
            out["frame_shape"] = list(s.screen_buffer.shape)
        return out

    def close(self):
        if self.game:
            self.game.close()
        return {"ok": True}


def main():
    b = Bridge()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.pop("cmd")
            fn = getattr(b, cmd)
            res = fn(**req)
        except Exception as e:  # noqa: BLE001 - report everything to the Rust side
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(res, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if cmd == "close":
            break


if __name__ == "__main__":
    main()
