"""
damper_downforce.py
Downforce and CLA from damper potentiometers, with per-section analysis.
Alex Aguilar Viel, 2026. MIT licence.

Inputs (numpy arrays, same length):
    t      time [s]
    v_kph  speed [km/h]
    ay_g   lateral acceleration [g]  (sign irrelevant)
    ax_g   longitudinal acceleration [g] (+ accelerating, - braking)
    pot    damper potentiometers [N, 4] in mm, order FL FR RL RR, compression = smaller value
Car description (dict):
    k_damper  spring rate at the damper [N/mm], length 4
    IR        installation ratio damper/wheel, length 4
    damper    function (v_mm_s, is_front) -> force [N] at the damper, + resists compression,
              without the static gas force (it is already in the zero reference)
    m, wb, tf, tr, h, wdf, hrc_f, hrc_r, anti_f, anti_r  for the corrections
The car must sit still for a few seconds in the log: that is the zero reference.
"""
import numpy as np
from scipy.signal import butter, filtfilt

RHO = 1.225


def lowpass(x, fs, fc):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def wheel_forces(t, pot, car, fc=8.0):
    """Extra load on each wheel relative to standstill [N]."""
    fs = 1 / np.median(np.diff(t))
    static = car.get("static_mask")
    x = lowpass(np.asarray(pot, float), fs, fc)
    dx = -(x - x[static].mean(0))                      # mm, + = compression
    vd = -np.gradient(dx, t, axis=0)                   # damper extension speed, mm/s
    fd = np.stack([car["damper"](vd[:, i], i < 2) for i in range(4)], 1)
    return car["IR"] * (car["k_damper"] * dx + fd)     # virtual work: F_wheel = IR * F_damper


def non_aero_load(ay_g, ax_g, car):
    """Part of the spring-load sum that weight transfer does NOT cancel."""
    g = 9.81
    ay, ax = np.abs(ay_g) * g, ax_g * g
    jack = car["m"] * ay * ((1 - car["wdf"]) * 2 * car["hrc_f"] / car["tf"] + car["wdf"] * 2 * car["hrc_r"] / car["tr"])
    anti = car["m"] * ax * car["h"] / car["wb"] * (car["anti_f"] - car["anti_r"])
    return -jack + anti


def downforce(t, v_kph, ay_g, ax_g, pot, car, correct=True):
    car = dict(car, static_mask=np.asarray(v_kph) < 0.5)
    s = wheel_forces(t, pot, car).sum(1)
    return s - non_aero_load(ay_g, ax_g, car) if correct else s


def fit_cla(v_kph, df):
    q = 0.5 * RHO * (np.asarray(v_kph) / 3.6) ** 2
    return float(np.sum(df * q) / np.sum(q * q))


def cla_by_section(v_kph, df, section_id, v_min=80):
    """One CLA per track section (e.g. from distance markers)."""
    out = {}
    for sid in np.unique(section_id):
        m = (section_id == sid) & (v_kph > v_min)
        if m.sum() > 20:
            out[int(sid)] = fit_cla(v_kph[m], df[m])
    return out


def bootstrap_ci(v_kph, df, mask, n=300, block=200, seed=0):
    idx = np.where(mask)[0]
    blocks = np.array_split(idx, max(1, len(idx) // block))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        vals.append(fit_cla(v_kph[sel], df[sel]))
    return np.percentile(vals, [2.5, 97.5])
