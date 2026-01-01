import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# CONFIG
MODE = "perpendicular"     # "parallel" or "perpendicular"
SHOW_ANIMATION = True
SAVE_FIGS = False
FIG_PREFIX = "out"

# Helpers
def wrap_pi(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def hypot2(x: float, y: float) -> float:
    return math.hypot(x, y)


# Geometry: Oriented box + SAT
@dataclass(frozen=True)
class OrientedBox:
    cx: float
    cy: float
    yaw: float
    length: float
    width: float

    def corners(self) -> List[Tuple[float, float]]:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        hl = self.length / 2.0
        hw = self.width / 2.0
        pts_local = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
        pts = []
        for lx, ly in pts_local:
            x = self.cx + lx * c - ly * s
            y = self.cy + lx * s + ly * c
            pts.append((x, y))
        return pts


def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def _project(axis, poly):
    ax, ay = axis
    mn, mx = float("inf"), -float("inf")
    for x, y in poly:
        p = _dot(ax, ay, x, y)
        mn = min(mn, p)
        mx = max(mx, p)
    return mn, mx


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def sat_intersect(poly1, poly2) -> bool:
    def normals(poly):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = -ey, ex
            n = math.hypot(nx, ny)
            if n == 0:
                continue
            yield (nx / n, ny / n)

    for axis in list(normals(poly1)) + list(normals(poly2)):
        if not _overlap(_project(axis, poly1), _project(axis, poly2)):
            return False
    return True


def point_in_oriented_box(px, py, box: OrientedBox) -> bool:
    # transform point to box frame
    c = math.cos(-box.yaw)
    s = math.sin(-box.yaw)
    dx = px - box.cx
    dy = py - box.cy
    x = dx * c - dy * s
    y = dx * s + dy * c
    return (abs(x) <= box.length / 2.0) and (abs(y) <= box.width / 2.0)


# Vehicle / Scenario
@dataclass
class Vehicle:
    # Ackermann-equivalent kinematic bicycle parameters
    L: float = 2.7
    length: float = 4.6
    width: float = 1.9
    delta_max: float = math.radians(30)
    safety_margin: float = 0.10


@dataclass
class Scenario:
    name: str
    bounds: Tuple[float, float, float, float]
    obstacles: List[OrientedBox]
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    slot: Optional[OrientedBox] = None
    wall: Optional[OrientedBox] = None   # perpendicular: back wall
    ceiling: Optional[OrientedBox] = None  # parallel: the "upper wall"


def vehicle_box(v: Vehicle, x, y, yaw) -> OrientedBox:
    return OrientedBox(
        cx=x, cy=y, yaw=yaw,
        length=v.length + 2 * v.safety_margin,
        width=v.width + 2 * v.safety_margin
    )


def collision(v: Vehicle, sc: Scenario, x, y, yaw) -> bool:
    vb = vehicle_box(v, x, y, yaw).corners()
    for obs in sc.obstacles:
        if sat_intersect(vb, obs.corners()):
            return True
    return False


def in_bounds(sc: Scenario, x, y) -> bool:
    xmin, xmax, ymin, ymax = sc.bounds
    return xmin <= x <= xmax and ymin <= y <= ymax


def car_fully_in_slot(v: Vehicle, sc: Scenario, x, y, yaw) -> bool:
    if sc.slot is None:
        return False
    corners = OrientedBox(x, y, yaw, v.length, v.width).corners()
    return all(point_in_oriented_box(cx, cy, sc.slot) for (cx, cy) in corners)


# Scenarios
def make_parallel_scenario(v: Vehicle) -> Scenario:
    """
    侧方停车：两台灰车 + 上方墙壁（距灰车 0.3m，长度大于三个车位）
    车位在两车之间（绿色框）。
    """
    bounds = (-6.0, 20.0, -8.0, 10.0)

    car_L = v.length
    car_W = v.width
    cy = 2.2

    # 两车留足净空（可停入）
    gap = 6.4  # 净空（>= 车长 + 少量余量）
    rear_cx = 2.0
    front_cx = rear_cx + car_L + gap

    obs_rear = OrientedBox(cx=rear_cx, cy=cy, yaw=0.0, length=car_L, width=car_W)
    obs_front = OrientedBox(cx=front_cx, cy=cy, yaw=0.0, length=car_L, width=car_W)

    slot_cx = (rear_cx + car_L / 2 + front_cx - car_L / 2) / 2.0
    slot = OrientedBox(cx=slot_cx, cy=cy, yaw=0.0, length=gap, width=2.7)

    # 墙壁：在灰车上方 0.3m，长度 > 3 个车位总长
    wall_y = cy + car_W / 2 + 0.3 + 0.18
    ceiling = OrientedBox(cx=slot_cx + 0.0, cy=wall_y, yaw=0.0, length=3 * (car_L + gap) + 10.0, width=0.35)

    # 起点：右下，朝向 x 正方向
    start = (slot_cx + 8.0, -1.0, 0.0)
    goal = (slot_cx, cy, 0.0)

    return Scenario(
        name="Parallel Parking",
        bounds=bounds,
        obstacles=[obs_front, obs_rear, ceiling],
        start=start,
        goal=goal,
        slot=slot,
        ceiling=ceiling
    )


def make_perpendicular_scenario(v: Vehicle) -> Scenario:
    """
    倒车入库：两侧障碍车 + 顶部墙(后方墙)；绿色框为车位。
    左右车与车位要拉开间距，否则根本无解。
    """
    bounds = (-10.0, 20.0, -10.0, 18.0) 

    # 车位中心
    slot = OrientedBox(cx=2.0, cy=8.4, yaw=math.pi / 2, length=5.8, width=3.2)

    # 左右障碍车（与车位留出足够侧向间隙）
    side_gap = 1.0
    left_x = slot.cx - (slot.width / 2 + v.width / 2 + side_gap)
    right_x = slot.cx + (slot.width / 2 + v.width / 2 + side_gap)

    # 障碍车停在车位“上方区域”，不要贴着车位边
    car_cy = slot.cy - 0.1
    left_car = OrientedBox(cx=left_x, cy=car_cy, yaw=math.pi / 2, length=v.length, width=v.width)
    right_car = OrientedBox(cx=right_x, cy=car_cy, yaw=math.pi / 2, length=v.length, width=v.width)

    # 后方墙（车位更上方）
    wall = OrientedBox(cx=slot.cx, cy=slot.cy + slot.length / 2 + 1.2, yaw=0.0, length=30.0, width=0.35)

    # 起点：右下方，略带角度
    start = (10.0, -0.5, math.radians(-40.0))
    goal = (slot.cx, slot.cy, slot.yaw)

    return Scenario(
        name="Perpendicular Parking",
        bounds=bounds,
        obstacles=[left_car, right_car, wall],
        start=start,
        goal=goal,
        slot=slot,
        wall=wall
    )

# Reference path planning
def gen_arc(x0, y0, yaw0, R, dtheta, n=160):
    thetas = np.linspace(0.0, dtheta, n)
    xs, ys, yaws = [], [], []
    for th in thetas:
        yaw = wrap_pi(yaw0 + th)
        x = x0 + R * (math.sin(yaw0 + th) - math.sin(yaw0))
        y = y0 - R * (math.cos(yaw0 + th) - math.cos(yaw0))
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
    return np.array(xs), np.array(ys), np.array(yaws)


def make_reference(sc: Scenario, v: Vehicle, dt=0.05):
    #生成参考路径：用于“模糊控制跟踪”
    x0, y0, yaw0 = sc.start
    xg, yg, yawg = sc.goal

    if "Parallel" in sc.name:
        Rmin = v.L / math.tan(v.delta_max)

        # 经验角度（可停进框的前提下再靠PSO调参数）
        d1 = -math.radians(35)   # swing
        d2 = math.radians(80)    # cut-in

        x1, y1, yaw1 = gen_arc(x0, y0, yaw0, R=Rmin, dtheta=d1, n=140)
        x2, y2, yaw2 = gen_arc(x1[-1], y1[-1], yaw1[-1], R=-Rmin, dtheta=d2, n=240)

        n3 = 140
        x3 = np.linspace(x2[-1], xg, n3)
        y3 = np.linspace(y2[-1], yg, n3)
        yaw3 = np.full(n3, yawg)

        x_ref = np.concatenate([x1, x2, x3])
        y_ref = np.concatenate([y1, y2, y3])
        yaw_ref = np.concatenate([yaw1, yaw2, yaw3])
        return x_ref, y_ref, yaw_ref, dt

    # Perpendicular: approach -> reverse in
    # approach point：位于车位口前下方
    if sc.slot is None:
        raise ValueError("Perpendicular scenario must have slot")

    # 车位口附近的“接近点”：位于 slot 中心下方
    ax = sc.slot.cx + 3.5
    ay = sc.slot.cy - sc.slot.length / 2 - 2.8
    ayaw = math.radians(-20.0)

    # 分两段：start -> approach (平滑曲线)；approach -> goal (反向入库参考)
    T1 = 7.0
    N1 = int(T1 / dt)
    s = np.linspace(0, 1, N1)
    x1 = x0 + (ax - x0) * (0.60 * s + 0.40 * s**2)
    y1 = y0 + (ay - y0) * (0.85 * s + 0.15 * s**2)
    yaw1 = np.linspace(yaw0, ayaw, N1)

    T2 = 9.0
    N2 = int(T2 / dt)
    s2 = np.linspace(0, 1, N2)
    x2 = ax + (xg - ax) * (s2**1.1)
    y2 = ay + (yg - ay) * (s2**1.05)
    yaw2 = np.linspace(ayaw, yawg, N2)

    x_ref = np.concatenate([x1, x2])
    y_ref = np.concatenate([y1, y2])
    yaw_ref = np.concatenate([yaw1, yaw2])
    return x_ref, y_ref, yaw_ref, dt

# Fuzzy controller
def tri_mf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def fuzzy_5(x):
    return {
        "NB": tri_mf(x, -1.5, -1.0, -0.5),
        "NS": tri_mf(x, -1.0, -0.5,  0.0),
        "Z":  tri_mf(x, -0.5,  0.0,  0.5),
        "PS": tri_mf(x,  0.0,  0.5,  1.0),
        "PB": tri_mf(x,  0.5,  1.0,  1.5),
    }


# 规则表（误差=行，航向=列 -> 输出）
RULE = {
    "NB": {"NB": "PB", "NS": "PB", "Z": "PB", "PS": "PS", "PB": "Z"},
    "NS": {"NB": "PB", "NS": "PS", "Z": "PS", "PS": "Z",  "PB": "NS"},
    "Z":  {"NB": "PS", "NS": "PS", "Z": "Z",  "PS": "NS", "PB": "NS"},
    "PS": {"NB": "Z",  "NS": "NS", "Z": "NS", "PS": "NS", "PB": "NB"},
    "PB": {"NB": "NS", "NS": "NB", "Z": "NB", "PS": "NB", "PB": "NB"},
}
OUT_VAL = {"NB": -1.0, "NS": -0.5, "Z": 0.0, "PS": 0.5, "PB": 1.0}


def nearest_index(x_ref, y_ref, x, y, start_i=0):
    dx = x_ref[start_i:] - x
    dy = y_ref[start_i:] - y
    i = int(np.argmin(dx * dx + dy * dy))
    return start_i + i

# Simulation + scoring
# params = [k_lat, k_yaw, k_delta, alpha, lookahead]
def simulate_and_score(params, v: Vehicle, sc: Scenario, return_traj=False, verbose_stop=False):
    k_lat, k_yaw, k_delta, alpha, lookahead = params
    lookahead = int(round(lookahead))

    # basic checks
    if k_lat <= 0 or k_yaw <= 0 or k_delta <= 0:
        return (1e9, None) if return_traj else 1e9
    if not (0.0 <= alpha <= 0.98):
        return (1e9, None) if return_traj else 1e9
    if not (2 <= lookahead <= 80):
        return (1e9, None) if return_traj else 1e9

    x_ref, y_ref, yaw_ref, dt = make_reference(sc, v, dt=0.05)

    x, y, yaw = sc.start
    xg, yg, yawg = sc.goal

    delta = 0.0
    i_ref = 0

    traj = []
    cost = 0.0
    steer_energy = 0.0
    PEN_COLL = 3e5
    PEN_OOB = 2e5

    # 任务区分的“形状化奖励”
    is_parallel = ("Parallel" in sc.name)

    # Perpendicular 用到 approach 点的 shaping：让它先“接近车位口”
    if (not is_parallel) and (sc.slot is not None):
        ax = sc.slot.cx + 3.5
        ay = sc.slot.cy - sc.slot.length / 2 - 2.8
    else:
        ax, ay = xg, yg

    for step in range(len(x_ref)):
        # ===== 参考目标点 =====
        i_ref = nearest_index(x_ref, y_ref, x, y, start_i=i_ref)
        i_tgt = min(i_ref + lookahead, len(x_ref) - 1)
        xt, yt, yawt = x_ref[i_tgt], y_ref[i_tgt], yaw_ref[i_tgt]

        ex = xt - x
        ey = yt - y

        e_long = math.cos(yaw) * ex + math.sin(yaw) * ey
        e_lat = -math.sin(yaw) * ex + math.cos(yaw) * ey
        e_yaw = wrap_pi(yawt - yaw)

        # ===== 末端“停正”模式：仅在接近车位中心时触发 =====
        dist_goal = hypot2(x - xg, y - yg)
        if dist_goal < (0.9 if is_parallel else 1.2):
            # 终端：强制靠近目标姿态
            Kpsi = 2.2 if is_parallel else 1.8
            delta_des = clamp(Kpsi * wrap_pi(yawg - yaw), -v.delta_max, v.delta_max)
            delta = 0.85 * delta + 0.15 * delta_des
            delta = clamp(delta, -v.delta_max, v.delta_max)

            # 慢速倒车（倒库/侧方都按倒车执行）
            v_cmd = -0.25 * math.tanh(1.2 * dist_goal)
        else:
            # ===== 模糊控制：输出增量 -> 转角 =====
            en = clamp(k_lat * e_lat, -1.0, 1.0)
            ywn = clamp(k_yaw * e_yaw, -1.0, 1.0)
            me = fuzzy_5(en)
            myw = fuzzy_5(ywn)

            num, deno = 0.0, 0.0
            for le, we in me.items():
                if we <= 1e-9:
                    continue
                for ly, wy in myw.items():
                    if wy <= 1e-9:
                        continue
                    w_rule = min(we, wy)
                    out_label = RULE[le][ly]
                    z = OUT_VAL[out_label]
                    num += w_rule * z
                    deno += w_rule
            u_norm = 0.0 if deno == 0 else num / deno

            # 增量映射：k_delta
            delta_cmd = delta + k_delta * u_norm
            delta = alpha * delta + (1 - alpha) * delta_cmd
            delta = clamp(delta, -v.delta_max, v.delta_max)

            # 倒车速度：离目标越远越快，靠近减速
            v_cmd = -0.85 * math.tanh(0.9 * dist_goal)

        # ===== Kinematic bicycle (Ackermann-equivalent) =====
        x += v_cmd * math.cos(yaw) * dt
        y += v_cmd * math.sin(yaw) * dt
        yaw = wrap_pi(yaw + (v_cmd / v.L) * math.tan(delta) * dt)

        traj.append((x, y, yaw))
        if len(traj) >= 2:
            dpsi = wrap_pi(traj[-1][2] - traj[-2][2])
            steer_energy += dpsi * dpsi

        # ===== hard stop conditions =====
        if not in_bounds(sc, x, y):
            if verbose_stop:
                print("[STOP] out of bounds at step", step, "x,y=", x, y)
            cost += PEN_OOB
            break

        if collision(v, sc, x, y, yaw):
            if verbose_stop:
                print("[STOP] collision: car at step", step, "x,y,yaw=", x, y, yaw)
            cost += PEN_COLL
            break

        # ===== shaped running cost (two different designs) =====
        if is_parallel:
            # 侧方：更强调“进框 + 航向”
            cost += 4.0 * (e_lat ** 2) + 1.0 * (e_long ** 2) + 2.5 * (e_yaw ** 2)
        else:
            # 倒库：先到 approach 点（让它别远离车位），再强调最终姿态
            d_app = hypot2(x - ax, y - ay)
            cost += 1.8 * (e_lat ** 2) + 0.8 * (e_long ** 2) + 1.2 * (e_yaw ** 2)
            cost += 0.04 * (d_app ** 2)  # 轻量 shaping：促使先接近车位口

        cost += 10.0 * steer_energy

        # ===== success stop: FULL in slot + yaw aligned =====
        yaw_err = abs(wrap_pi(yaw - yawg))
        if car_fully_in_slot(v, sc, x, y, yaw) and yaw_err < math.radians(4.0) and abs(delta) < math.radians(3.0):
            # 成功给强奖励
            cost -= 2.0e5
            break

    # ===== terminal cost: enforce "fully inside slot" =====
    pos_err = hypot2(x - xg, y - yg)
    yaw_err = abs(wrap_pi(yaw - yawg))

    # 车是否完全进框
    inside = car_fully_in_slot(v, sc, x, y, yaw)

    if is_parallel:
        # 强制停正：没进框大罚
        if not inside:
            cost += 2.0e5 + 2.0e4 * (pos_err ** 2) + 8.0e4 * (yaw_err ** 2)
        else:
            cost += 8.0e3 * (pos_err ** 2) + 1.2e4 * (yaw_err ** 2)
    else:
        # 倒库：允许先接近再微调，但最终必须进框
        if not inside:
            cost += 1.5e5 + 1.5e4 * (pos_err ** 2) + 4.0e4 * (yaw_err ** 2)
        else:
            cost += 1.2e4 * (pos_err ** 2) + 8.0e3 * (yaw_err ** 2)

    traj_np = np.array(traj) if len(traj) > 0 else np.zeros((0, 3))
    return (cost, traj_np) if return_traj else cost

# Visualization
def draw_oriented_box(ax, box: OrientedBox, **kwargs):
    poly = Polygon(box.corners(), closed=True, **kwargs)
    ax.add_patch(poly)
    return poly


def draw_vehicle(ax, v: Vehicle, x, y, yaw, facecolor="orange", edgecolor="black", alpha=0.35):
    box = OrientedBox(cx=x, cy=y, yaw=yaw, length=v.length, width=v.width)
    return draw_oriented_box(ax, box, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)


def plot_scene(ax, v: Vehicle, sc: Scenario):
    xmin, xmax, ymin, ymax = sc.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    for obs in sc.obstacles:
        draw_oriented_box(ax, obs, facecolor="gray", edgecolor="black", alpha=0.55)

    if sc.slot is not None:
        draw_oriented_box(ax, sc.slot, facecolor="none", edgecolor="green", linewidth=2.2)

    ax.scatter([sc.start[0]], [sc.start[1]], c="blue", s=60, label="start")
    ax.scatter([sc.goal[0]], [sc.goal[1]], c="red", s=60, label="goal")


def plot_exploration(sc: Scenario, v: Vehicle, all_trajs, title="Exploration trajectories"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_scene(ax, v, sc)

    for tr in all_trajs:
        if tr.shape[0] == 0:
            continue
        ax.plot(tr[:, 0], tr[:, 1], linewidth=0.8, alpha=0.12)

    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(f"{FIG_PREFIX}_exploration.png", dpi=220)
    plt.show()


def plot_final_path(sc: Scenario, v: Vehicle, traj, x_ref=None, y_ref=None, title="Final path"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_scene(ax, v, sc)

    if x_ref is not None and y_ref is not None:
        ax.plot(x_ref, y_ref, "r--", label="ref(path)")
    ax.plot(traj[:, 0], traj[:, 1], "b", linewidth=2.4, label="track")

    k = max(1, len(traj) // 18)
    for i in range(0, len(traj), k):
        draw_vehicle(ax, v, traj[i, 0], traj[i, 1], traj[i, 2], alpha=0.18)

    # final car outline (black edge)
    if len(traj) > 0:
        draw_vehicle(ax, v, traj[-1, 0], traj[-1, 1], traj[-1, 2], alpha=0.45)

    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(f"{FIG_PREFIX}_final.png", dpi=220)
    plt.show()


def animate_final(sc: Scenario, v: Vehicle, traj, x_ref=None, y_ref=None, title="Parking animation"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_scene(ax, v, sc)

    if x_ref is not None and y_ref is not None:
        ax.plot(x_ref, y_ref, "r--", label="ref(path)")
    ax.plot(traj[:, 0], traj[:, 1], "b", linewidth=2.4, label="track")

    car_patch = None
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()

    for i in range(len(traj)):
        if car_patch is not None:
            car_patch.remove()

        car_box = OrientedBox(cx=traj[i, 0], cy=traj[i, 1], yaw=traj[i, 2], length=v.length, width=v.width)
        car_patch = Polygon(car_box.corners(), closed=True, facecolor="orange", edgecolor="black", alpha=0.45)
        ax.add_patch(car_patch)

        plt.pause(0.02)

    plt.show()

# PSO
def pso_optimize(v: Vehicle, sc: Scenario, iters=35, swarm=28, seed=3):
    random.seed(seed)
    np.random.seed(seed)

    # params = [k_lat, k_yaw, k_delta, alpha, lookahead]
    lb = np.array([0.20, 0.60, 0.04, 0.00, 5.0], dtype=float)
    ub = np.array([3.00, 4.00, 1.20, 0.95, 60.0], dtype=float)
    dim = len(lb)

    X = lb + (ub - lb) * np.random.rand(swarm, dim)
    Vv = np.zeros((swarm, dim))

    # seed particle
    X[0] = np.array([1.0, 2.0, 0.35, 0.7, 20.0], dtype=float)

    pbest = X.copy()
    pbest_val = np.array([simulate_and_score(p, v, sc) for p in pbest])

    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    w = 0.72
    c1 = 1.4
    c2 = 1.4

    hist = []
    all_trajs = []

    for it in range(iters):
        r1 = np.random.rand(swarm, dim)
        r2 = np.random.rand(swarm, dim)
        Vv = w * Vv + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + Vv
        X = np.clip(X, lb, ub)

        vals = np.array([simulate_and_score(p, v, sc) for p in X])

        improved = vals < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = vals[improved]

        gbest_idx = int(np.argmin(pbest_val))
        if pbest_val[gbest_idx] < gbest_val:
            gbest_val = float(pbest_val[gbest_idx])
            gbest = pbest[gbest_idx].copy()

        hist.append(gbest_val)
        print(f"[{sc.name}] iter {it+1:02d}/{iters} best_cost={gbest_val:.2f} best={gbest}")

        # sample trajectories for exploration
        sample_n = min(6, swarm)
        sample_idx = np.random.choice(np.arange(swarm), size=sample_n, replace=False)
        for idx in sample_idx:
            _, tr = simulate_and_score(X[idx], v, sc, return_traj=True)
            all_trajs.append(tr)

    return gbest, gbest_val, hist, all_trajs


def plot_history(hist, title):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(hist)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("iter")
    plt.ylabel("best cost")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(f"{FIG_PREFIX}_history.png", dpi=220)
    plt.show()

# Main
def main():
    v = Vehicle()

    if MODE == "parallel":
        sc = make_parallel_scenario(v)
    else:
        sc = make_perpendicular_scenario(v)

    start_coll = collision(v, sc, sc.start[0], sc.start[1], sc.start[2])
    print("MODE:", MODE, "| start:", sc.start, "| start collision:", start_coll)

    best, best_val, hist, all_trajs = pso_optimize(v, sc, iters=35, swarm=28, seed=3)
    print("Final best:", best, "cost:", best_val)

    plot_history(hist, f"{sc.name} - PSO best cost")
    plot_exploration(sc, v, all_trajs, title=f"{sc.name} - Exploration (sampled trajectories)")

    cost, final_traj = simulate_and_score(best, v, sc, return_traj=True, verbose_stop=True)
    x_ref, y_ref, yaw_ref, dt = make_reference(sc, v, dt=0.05)

    plot_final_path(
        sc, v, final_traj,
        x_ref=x_ref, y_ref=y_ref,
        title=(
            f"{sc.name}\n"
            f"k_lat={best[0]:.2f}, k_yaw={best[1]:.2f}, k_delta={best[2]:.2f}, alpha={best[3]:.2f}, Ld={int(round(best[4]))}\n"
            f"final_cost={cost:.1f}"
        )
    )

    if SHOW_ANIMATION:
        animate_final(sc, v, final_traj, x_ref=x_ref, y_ref=y_ref, title=f"{sc.name} - Animation")


if __name__ == "__main__":
    main()
