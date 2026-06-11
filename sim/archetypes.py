"""Per-archetype event generators.

Each generator returns (user_record: dict, events: EventBuffer).
Timestamps are epoch seconds (int). Networks are (ip, asn, asn_type, country).
"""
import numpy as np

from . import prompts
from .world import World, UA_DESKTOP, UA_MOBILE, UA_BOT

DAY = 86400

EVENT_COLS = ["ts", "ip", "asn", "asn_type", "country", "device_fp",
              "user_agent", "endpoint", "prompt_cluster", "prompt_text",
              "tokens_used", "success"]

ENDPOINTS = ["/v1/chat", "/v1/completions", "/v1/embeddings"]

TZ_CHOICES = [-8, -8, -7, -6, -6, -5, -5, -5, -5, 0, 1, 1, 5.5, 8, 9]
TZ_COUNTRY = {-8: "US", -7: "US", -6: "US", -5: "US", 0: "GB", 1: "DE",
              5.5: "IN", 8: "SG", 9: "KR"}


class EventBuffer:
    __slots__ = ("cols",)

    def __init__(self):
        self.cols = {k: [] for k in EVENT_COLS}

    def add(self, ts, net, dev, ua, ep, cluster, text, tokens, ok):
        c = self.cols
        c["ts"].append(int(ts))
        c["ip"].append(net[0]); c["asn"].append(net[1])
        c["asn_type"].append(net[2]); c["country"].append(net[3])
        c["device_fp"].append(dev); c["user_agent"].append(ua)
        c["endpoint"].append(ep); c["prompt_cluster"].append(cluster)
        c["prompt_text"].append(text); c["tokens_used"].append(int(tokens))
        c["success"].append(bool(ok))

    def __len__(self):
        return len(self.cols["ts"])


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _human_sessions(rng, t0, t1, tz, peaks, p_active, sess_lambda):
    """Diurnal session start times between t0 and t1 (local peak hours -> UTC)."""
    out = []
    for d in range(int(t0 // DAY), int(t1 // DAY) + 1):
        if rng.random() > p_active:
            continue
        for _ in range(1 + rng.poisson(sess_lambda)):
            local_h = _pick(rng, peaks) + rng.normal(0, 1.8)
            ts = d * DAY + ((local_h - tz) % 24) * 3600 + rng.uniform(0, 600)
            if t0 <= ts < t1:
                out.append(ts)
    out.sort()
    return out


def _topic_profile(rng, n_topics):
    """Dirichlet interest profile over a random subset of normal topics."""
    k = int(rng.integers(*n_topics))
    ids = rng.choice(len(prompts.TOPICS), size=k, replace=False)
    w = rng.dirichlet(np.full(k, 0.6))
    return ids, w


def _endpoint(rng):
    r = rng.random()
    return ENDPOINTS[0] if r < 0.85 else (ENDPOINTS[1] if r < 0.95 else ENDPOINTS[2])


# ---------------------------------------------------------------------------
# Normal users (and the pre-takeover phase of token thieves)
# ---------------------------------------------------------------------------
def _normal_core(rng, world: World, cfg, sub=None, hard_end=None):
    if sub is None:
        sub = rng.choice(["regular", "power_user", "campus", "vpn", "corporate"],
                         p=[0.80, 0.06, 0.06, 0.04, 0.04])
    tz = _pick(rng, TZ_CHOICES)
    country = TZ_COUNTRY[tz]
    end = hard_end if hard_end is not None else cfg.end_ts

    # account age: mostly established, some created mid-window
    if rng.random() < 0.10:
        created = int(rng.uniform(cfg.start_ts, cfg.end_ts - DAY))
    else:
        created = int(cfg.start_ts - rng.uniform(5, 400) * DAY)
    t0 = max(cfg.start_ts, created)

    # network contexts
    home = world.campus() if sub == "campus" else world.residential(country)
    work = world.corporate() if sub == "corporate" else world.residential(country)
    carrier = world.mobile_carrier(country)

    devices = [world.device_fp() for _ in range(int(rng.integers(1, 3)))]
    uas = [_pick(rng, UA_DESKTOP if i == 0 else UA_MOBILE) for i in range(len(devices))]

    if sub == "power_user":
        p_active, sess_lambda = 0.95, 2.5
        ev_mean_extra, gap_mu, gap_sigma = 12, np.log(12), 0.9
        topic_ids, weights = _topic_profile(rng, (1, 3))
        dup_p = 0.45  # re-runs near-identical prompts: bot-like hard negative
    else:
        p_active = rng.beta(5, 3)
        sess_lambda = rng.uniform(0.4, 1.2)
        ev_mean_extra, gap_mu, gap_sigma = 5, np.log(45), 1.1
        topic_ids, weights = _topic_profile(rng, (4, 12))
        dup_p = 0.04

    peaks = list(rng.choice([8, 10, 13, 15, 19, 21, 23], size=int(rng.integers(2, 4)),
                            replace=False))

    ev = EventBuffer()
    last_text = None
    for s_ts in _human_sessions(rng, t0, end, tz, peaks, p_active, sess_lambda):
        r = rng.random()
        if sub == "vpn":
            net = world.datacenter()
        elif r < 0.50:
            net = home
        elif r < 0.80:
            net = work
        else:
            net = world.mobile_ip(carrier)
        di = int(rng.integers(len(devices)))
        n_ev = 1 + rng.geometric(1.0 / ev_mean_extra)
        t = s_ts
        for _ in range(n_ev):
            if last_text is not None and rng.random() < dup_p:
                cluster, text = last_cluster, last_text
            else:
                cluster, text = prompts.normal_prompt(rng, topic_ids, weights)
            last_cluster, last_text = cluster, text
            ev.add(t, net, devices[di], uas[di], _endpoint(rng), cluster, text,
                   rng.lognormal(np.log(350), 0.8), rng.random() > 0.02)
            t += max(2.0, rng.lognormal(gap_mu, gap_sigma))
            if t >= end:
                break

    user = {
        "label": "normal", "subtype": sub, "ring_id": None,
        "created_at": created, "signup_country": country,
        "payment_hash": world.payment_hash() if rng.random() < 0.6 else None,
        "username": world.username(), "tz_offset": float(tz), "takeover_ts": None,
        "_devices": devices, "_home": home,
    }
    return user, ev


def gen_normal(rng, world, cfg):
    user, ev = _normal_core(rng, world, cfg)
    user.pop("_devices"), user.pop("_home")
    return user, ev


# ---------------------------------------------------------------------------
# Spam bots: regular timing, low prompt entropy, shared datacenter infra
# ---------------------------------------------------------------------------
def make_bot_farm(rng, world: World, cfg):
    return {
        "ips": [world.datacenter() for _ in range(int(rng.integers(3, 11)))],
        "devices": [world.device_fp() for _ in range(int(rng.integers(1, 6)))],
        "ua": _pick(rng, UA_BOT),
        "clusters": list(rng.choice(len(prompts.SPAM),
                                    size=int(rng.integers(1, 3)), replace=False)),
        "period": rng.uniform(180, 1200),       # seconds between requests
        "endpoint": _pick(rng, ENDPOINTS[:2]),
    }


def gen_spam_bot(rng, world, cfg, farm, ring_id):
    created = int(cfg.start_ts - rng.uniform(0, 30) * DAY)
    period = farm["period"] * rng.uniform(0.9, 1.1)
    jitter = period * rng.uniform(0.02, 0.08)
    duty_start = rng.uniform(0, 24)
    duty_len = rng.uniform(6, 24)               # hours active per day

    ev = EventBuffer()
    dev = _pick(rng, farm["devices"])
    t = max(cfg.start_ts, created) + rng.uniform(0, period)
    day_net = _pick(rng, farm["ips"])
    cur_day = -1
    while t < cfg.end_ts:
        d = int(t // DAY)
        if d != cur_day:
            cur_day, day_net = d, _pick(rng, farm["ips"])
        h = (t % DAY) / 3600
        if (h - duty_start) % 24 < duty_len:
            cluster, text = prompts.spam_prompt(rng, _pick(rng, farm["clusters"]))
            ev.add(t, day_net, dev, farm["ua"], farm["endpoint"], cluster, text,
                   rng.lognormal(np.log(450), 0.3), rng.random() > 0.15)
        t += max(1.0, rng.normal(period, jitter))

    user = {
        "label": "spam_bot", "subtype": "farm_bot", "ring_id": ring_id,
        "created_at": created, "signup_country": farm["ips"][0][3],
        "payment_hash": None, "username": world.username(),
        "tz_offset": 0.0, "takeover_ts": None,
    }
    return user, ev


# ---------------------------------------------------------------------------
# Account farmers: burst-created rings, shared devices/payment, dormant->active
# ---------------------------------------------------------------------------
def make_farmer_ring(rng, world: World, cfg, size):
    burst = int(cfg.start_ts - rng.uniform(2, 25) * DAY)
    gaps = rng.exponential(90, size).cumsum()
    return {
        "created": [int(burst + g) for g in gaps],
        "devices": [world.device_fp() for _ in range(int(rng.integers(1, 5)))],
        "payments": [world.payment_hash() for _ in range(int(rng.integers(1, 4)))],
        "signup_net": world.datacenter() if rng.random() < 0.6 else world.residential(),
        "base_name": world.username().split("_")[0] + _pick(rng, ["x", "_", "99", "hq"]),
        "dormant_days": rng.uniform(2, 14),
        "clusters": list(rng.choice(len(prompts.SPAM), size=1)),
        "topic_ids": rng.choice(len(prompts.TOPICS), size=2, replace=False),
    }


def gen_farmer(rng, world, cfg, ring, ring_id, idx):
    created = ring["created"][idx]
    activated = created + (ring["dormant_days"] + rng.uniform(-1, 3)) * DAY
    ev = EventBuffer()
    dev = _pick(rng, ring["devices"])
    ua = _pick(rng, UA_BOT if rng.random() < 0.5 else UA_DESKTOP)

    if rng.random() > 0.30:  # 30% never activate inside the window
        t0 = max(cfg.start_ts, activated)
        weights = np.array([0.7, 0.3])
        for s_ts in _human_sessions(rng, t0, cfg.end_ts, 0,
                                    [9, 14, 20], 0.5, 0.5):
            net = ring["signup_net"] if rng.random() < 0.7 else world.datacenter()
            t = s_ts
            for _ in range(int(rng.integers(3, 9))):
                if rng.random() < 0.5:
                    cluster, text = prompts.spam_prompt(rng, ring["clusters"][0])
                else:
                    cluster, text = prompts.normal_prompt(rng, ring["topic_ids"], weights)
                ev.add(t, net, dev, ua, "/v1/chat", cluster, text,
                       rng.lognormal(np.log(400), 0.4), rng.random() > 0.05)
                t += max(2.0, rng.lognormal(np.log(30), 0.5))

    user = {
        "label": "account_farmer", "subtype": "farm_account", "ring_id": ring_id,
        "created_at": created, "signup_country": ring["signup_net"][3],
        "payment_hash": _pick(rng, ring["payments"]),
        "username": world.farm_username(ring["base_name"], idx),
        "tz_offset": 0.0, "takeover_ts": None,
    }
    return user, ev


# ---------------------------------------------------------------------------
# Prompt sprayers: paraphrase-cycling jailbreak attempts in bursts
# ---------------------------------------------------------------------------
def gen_sprayer(rng, world, cfg):
    # half blend in light normal activity as cover
    if rng.random() < 0.5:
        user, ev = _normal_core(rng, world, cfg, sub="regular")
        devices = user.pop("_devices"); user.pop("_home")
        # thin the cover traffic
        keep = max(1, len(ev) // 4)
        for k in ev.cols:
            ev.cols[k] = ev.cols[k][:keep]
    else:
        user = {
            "created_at": int(cfg.start_ts - rng.uniform(1, 90) * DAY),
            "signup_country": "US", "payment_hash": None,
            "username": world.username(), "tz_offset": 0.0,
        }
        ev = EventBuffer()
        devices = [world.device_fp() for _ in range(int(rng.integers(1, 3)))]

    n_days = int(rng.integers(2, 9))
    burst_days = rng.choice(cfg.days, size=min(n_days, cfg.days), replace=False)
    intents = list(range(len(prompts.JAILBREAK)))
    ua = _pick(rng, UA_DESKTOP)
    for bd in burst_days:
        for _ in range(int(rng.integers(1, 4))):
            rng.shuffle(intents)
            net = (world.datacenter() if rng.random() < 0.7
                   else world.residential())  # rotating residential proxies
            t = cfg.start_ts + bd * DAY + rng.uniform(0, DAY - 4000)
            for i in range(int(rng.integers(15, 60))):
                cluster, text = prompts.jailbreak_prompt(rng, intents[i % len(intents)])
                ev.add(t, net, _pick(rng, devices), ua, "/v1/chat", cluster, text,
                       rng.lognormal(np.log(120), 0.4), rng.random() < 0.30)
                t += max(2.0, rng.lognormal(np.log(20), 0.6))

    user.update({"label": "prompt_sprayer", "subtype": "sprayer",
                 "ring_id": None, "takeover_ts": None})
    return user, ev


# ---------------------------------------------------------------------------
# Token thieves: established normal account, then behavioral discontinuity
# ---------------------------------------------------------------------------
def gen_thief(rng, world, cfg):
    takeover = int(cfg.start_ts + rng.uniform(0.30, 0.85) * cfg.days * DAY)
    user, ev = _normal_core(rng, world, cfg, sub="regular", hard_end=takeover)
    user.pop("_devices")
    home = user.pop("_home")

    # attacker phase: new device, foreign datacenter IP, extraction prompts
    atk_dev = world.device_fp()
    atk_net = world.datacenter(exclude_country=home[3])
    atk_ua = _pick(rng, UA_BOT)
    t = takeover + rng.uniform(60, 3600)
    burst_end = min(cfg.end_ts, takeover + rng.uniform(1, 3) * DAY)
    while t < burst_end:
        for _ in range(int(rng.integers(30, 120))):
            cluster, text = prompts.extract_prompt(rng)
            ev.add(t, atk_net, atk_dev, atk_ua, "/v1/chat", cluster, text,
                   rng.lognormal(np.log(1500), 0.5), rng.random() > 0.10)
            t += max(1.0, rng.lognormal(np.log(8), 0.5))
            if t >= burst_end:
                break
        t += rng.uniform(2, 5) * 3600  # pause between extraction sessions

    user.update({"label": "token_thief", "subtype": "ato",
                 "ring_id": None, "takeover_ts": takeover})
    return user, ev
