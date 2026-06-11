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

    # per-user nuisance params: token usage style and personal failure rate
    tok_mu = rng.normal(np.log(350), 0.5)
    tok_sigma = rng.uniform(0.5, 1.1)
    fail_p = rng.beta(1, 30)

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
                   rng.lognormal(tok_mu, tok_sigma), rng.random() > fail_p)
            t += max(2.0, rng.lognormal(gap_mu, gap_sigma))
            if t >= end:
                break

    user = {
        "label": "normal", "subtype": sub, "ring_id": None,
        "created_at": created, "signup_country": country,
        "payment_hash": world.payment_hash() if rng.random() < 0.6 else None,
        "username": world.username(), "tz_offset": float(tz), "takeover_ts": None,
        "evasion": 0.0,
        "_devices": devices, "_home": home, "_topics": (topic_ids, weights),
        "_tok": (tok_mu, tok_sigma), "_fail": fail_p,
    }
    return user, ev


def _strip(user):
    for k in ("_devices", "_home", "_topics", "_tok", "_fail"):
        user.pop(k, None)
    return user


def gen_normal(rng, world, cfg):
    user, ev = _normal_core(rng, world, cfg)
    return _strip(user), ev


# ---------------------------------------------------------------------------
# Spam bots: regular timing, low prompt entropy, shared datacenter infra
# ---------------------------------------------------------------------------
def make_bot_farm(rng, world: World, cfg, e=0.0):
    n_clusters = 2 if e < 0.5 else 4  # evasive farms rotate more campaigns
    return {
        "e": e,
        "ips": [world.datacenter() for _ in range(int(rng.integers(3, 11)))],
        "devices": [world.device_fp() for _ in range(int(rng.integers(1, 6)))],
        "ua": _pick(rng, UA_BOT),
        "clusters": list(rng.choice(len(prompts.SPAM),
                                    size=int(rng.integers(1, n_clusters + 1)),
                                    replace=False)),
        "period": rng.uniform(180, 1200) * (1 + 2 * e),  # evasive bots throttle
        "endpoint": _pick(rng, ENDPOINTS[:2]),
    }


def gen_spam_bot(rng, world, cfg, farm, ring_id):
    e = farm["e"]
    created = int(cfg.start_ts - rng.uniform(0, 30) * DAY)
    period = farm["period"] * rng.uniform(0.9, 1.1)
    # naive bots: tight metronome jitter; evasive: human-scale variance
    jitter_frac = 0.02 + 0.6 * e
    duty_start = rng.uniform(0, 24)
    duty_len = rng.uniform(6, 24) if e < 0.5 else rng.uniform(6, 14)  # fake sleep

    # evasive bots sometimes run their own device / browser UA / resi proxies
    own_dev = rng.random() < 0.6 * e
    dev = world.device_fp() if own_dev else _pick(rng, farm["devices"])
    ua = _pick(rng, UA_DESKTOP) if rng.random() < e else farm["ua"]
    resi_p = 0.7 * e   # daily residential-proxy rotation prob
    blend_p = 0.10 + 0.45 * e  # prob of a normal-looking prompt
    topic_ids = rng.choice(len(prompts.TOPICS), size=3, replace=False)
    weights = np.full(3, 1 / 3)
    # nuisance marginals converge to the normal population with evasion
    tok_mu = (1 - e) * np.log(450) + e * rng.normal(np.log(350), 0.5)
    tok_sigma = 0.3 + 0.7 * e * rng.uniform(0.5, 1.0)
    fail_p = (1 - e) * 0.15 + e * rng.beta(1, 30)
    ep = "/v1/chat" if rng.random() < e else farm["endpoint"]

    def emit(buf, t, net):
        if rng.random() < blend_p:
            cluster, text = prompts.normal_prompt(rng, topic_ids, weights)
        else:
            cluster, text = prompts.spam_prompt(rng, _pick(rng, farm["clusters"]))
        buf.add(t, net, dev, ua, ep, cluster, text,
                rng.lognormal(tok_mu, tok_sigma), rng.random() > fail_p)

    ev = EventBuffer()
    if rng.random() < e:
        # compromised/aged-account mode: a fully normal behavioral envelope
        # (often a purchased aged account) with spam payload injected and
        # occasional routing through shared farm infrastructure
        sub = "power_user" if rng.random() < 0.3 else "regular"
        nuser, ev = _normal_core(rng, world, cfg, sub=sub)
        created = nuser["created_at"]
        spam_frac = rng.uniform(0.2, 0.6)
        route_p = 0.25          # days routed via farm ips/devices (cost pressure)
        c = ev.cols
        day_routed = {}
        for i in range(len(ev)):
            if rng.random() < spam_frac:
                cluster, text = prompts.spam_prompt(rng, _pick(rng, farm["clusters"]))
                c["prompt_cluster"][i] = cluster
                c["prompt_text"][i] = text
            d = c["ts"][i] // DAY
            if d not in day_routed:
                day_routed[d] = rng.random() < route_p
            if day_routed[d]:
                ip, asn, asn_type, country = _pick(rng, farm["ips"])
                c["ip"][i], c["asn"][i] = ip, asn
                c["asn_type"][i], c["country"][i] = asn_type, country
                c["device_fp"][i] = dev
    else:
        # metronome mode: fixed period + jitter, duty-cycled
        t = max(cfg.start_ts, created) + rng.uniform(0, period)
        day_net = _pick(rng, farm["ips"])
        cur_day = -1
        while t < cfg.end_ts:
            d = int(t // DAY)
            if d != cur_day:
                cur_day = d
                day_net = (world.residential() if rng.random() < resi_p
                           else _pick(rng, farm["ips"]))
            h = (t % DAY) / 3600
            if (h - duty_start) % 24 < duty_len:
                emit(ev, t, day_net)
            t += max(1.0, rng.normal(period, period * jitter_frac))

    user = {
        "label": "spam_bot", "subtype": "farm_bot", "ring_id": ring_id,
        "created_at": created, "signup_country": farm["ips"][0][3],
        "payment_hash": None, "username": world.username(),
        "tz_offset": 0.0, "takeover_ts": None, "evasion": e,
    }
    return user, ev


# ---------------------------------------------------------------------------
# Account farmers: burst-created rings, shared devices/payment, dormant->active
# ---------------------------------------------------------------------------
def make_farmer_ring(rng, world: World, cfg, size, e=0.0):
    # evasive rings buy aged accounts / register far in advance
    burst = int(cfg.start_ts - rng.uniform(2, 25 + 300 * e) * DAY)
    # evasive rings stagger creation over days instead of a minutes-scale burst
    gaps = rng.exponential(90 * (1 + 30 * e), size).cumsum()
    n_dev = int(rng.integers(1, 5)) + int(e * size * 0.4)   # buy more devices
    n_pay = int(rng.integers(1, 4)) + int(e * size * 0.25)  # more stolen cards
    return {
        "e": e,
        "created": [int(burst + g) for g in gaps],
        "devices": [world.device_fp() for _ in range(n_dev)],
        "payments": [world.payment_hash() for _ in range(n_pay)],
        "signup_net": world.datacenter() if rng.random() < 0.6 - 0.4 * e
        else world.residential(),
        "base_name": world.username().split("_")[0] + _pick(rng, ["x", "_", "99", "hq"]),
        "dormant_days": rng.uniform(2, 14),
        "clusters": list(rng.choice(len(prompts.SPAM), size=1)),
        "topic_ids": rng.choice(len(prompts.TOPICS), size=4, replace=False),
    }


def gen_farmer(rng, world, cfg, ring, ring_id, idx):
    e = ring["e"]
    created = ring["created"][idx]
    activated = created + (ring["dormant_days"] + rng.uniform(-1, 3 + 10 * e)) * DAY
    ev = EventBuffer()
    dev = _pick(rng, ring["devices"])
    ua = _pick(rng, UA_BOT if rng.random() < 0.5 - 0.4 * e else UA_DESKTOP)
    # evasive accounts use individual usernames, not sequential ones
    username = (world.username() if rng.random() < e
                else world.farm_username(ring["base_name"], idx))
    spam_p = 0.5 - 0.3 * e
    weights = np.full(4, 0.25)
    tok_mu = rng.normal(np.log(380), 0.4)
    tok_sigma = rng.uniform(0.4, 1.0)
    fail_p = 0.02 + rng.beta(1, 30)

    if rng.random() > 0.30:  # 30% never activate inside the window
        t0 = max(cfg.start_ts, activated)
        for s_ts in _human_sessions(rng, t0, cfg.end_ts, 0,
                                    [9, 14, 20], 0.5, 0.5):
            r = rng.random()
            net = (ring["signup_net"] if r < 0.7 - 0.5 * e
                   else world.residential() if r < 0.7 else world.datacenter())
            t = s_ts
            for _ in range(int(rng.integers(3, 9))):
                if rng.random() < spam_p:
                    cluster, text = prompts.spam_prompt(rng, ring["clusters"][0])
                else:
                    cluster, text = prompts.normal_prompt(rng, ring["topic_ids"], weights)
                ev.add(t, net, dev, ua, "/v1/chat", cluster, text,
                       rng.lognormal(tok_mu, tok_sigma), rng.random() > fail_p)
                t += max(2.0, rng.lognormal(np.log(30), 0.5 + 0.6 * e))

    user = {
        "label": "account_farmer", "subtype": "farm_account", "ring_id": ring_id,
        "created_at": created, "signup_country": ring["signup_net"][3],
        "payment_hash": _pick(rng, ring["payments"]),
        "username": username,
        "tz_offset": 0.0, "takeover_ts": None, "evasion": e,
    }
    return user, ev


# ---------------------------------------------------------------------------
# Prompt sprayers: paraphrase-cycling jailbreak attempts in bursts
# ---------------------------------------------------------------------------
def gen_sprayer(rng, world, cfg, e=0.0):
    if e > 0.6:
        # low-and-slow mode: a fully normal envelope; a random subset of
        # events is swapped for paraphrased jailbreak attempts. timing and
        # volume are indistinguishable — only the content gives it away.
        user, ev = _normal_core(rng, world, cfg, sub="regular")
        _strip(user)
        n = len(ev)
        intents = list(range(len(prompts.JAILBREAK)))
        rng.shuffle(intents)
        n_jb = min(n, int(rng.uniform(30, 150)))
        c = ev.cols
        # success is untouched: a safety refusal is an HTTP 200, invisible
        # at the transport level — only content analysis can see this attack
        for j, i in enumerate(sorted(rng.choice(n, size=n_jb, replace=False))):
            cluster, text = prompts.jailbreak_prompt(rng, intents[j % len(intents)])
            c["prompt_cluster"][i] = cluster
            c["prompt_text"][i] = text
        user.update({"label": "prompt_sprayer", "subtype": "sprayer_slow",
                     "ring_id": None, "takeover_ts": None, "evasion": e})
        return user, ev

    # burst mode: cover traffic for some, dedicated throwaway accounts for rest
    if rng.random() < 0.5 + 0.5 * e:
        user, ev = _normal_core(rng, world, cfg, sub="regular")
        devices = user["_devices"]
        _strip(user)
        # naive keep little cover; evasive keep nearly all of it
        keep = max(1, int(len(ev) * (0.25 + 0.75 * e)))
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

    # evasive: fewer attempts per session, spread over more days, slower
    n_days = int(rng.integers(2, 9) + e * 12)
    burst_days = rng.choice(cfg.days, size=min(n_days, cfg.days), replace=False)
    intents = list(range(len(prompts.JAILBREAK)))
    ua = _pick(rng, UA_DESKTOP)
    per_session = (3 + int(9 * rng.random())) if e > 0.5 else int(rng.integers(15, 60))
    gap_mu = np.log(20 + 400 * e)
    for bd in burst_days:
        for _ in range(int(rng.integers(1, 4 - int(2 * e)))):
            rng.shuffle(intents)
            net = (world.datacenter() if rng.random() < 0.7 - 0.4 * e
                   else world.residential())  # rotating residential proxies
            t = cfg.start_ts + bd * DAY + rng.uniform(0, DAY - 4000)
            for i in range(per_session):
                cluster, text = prompts.jailbreak_prompt(rng, intents[i % len(intents)])
                ev.add(t, net, _pick(rng, devices), ua, "/v1/chat", cluster, text,
                       rng.lognormal(np.log(120), 0.4), rng.random() < 0.30)
                t += max(2.0, rng.lognormal(gap_mu, 0.6))

    user.update({"label": "prompt_sprayer", "subtype": "sprayer",
                 "ring_id": None, "takeover_ts": None, "evasion": e})
    return user, ev


# ---------------------------------------------------------------------------
# Token thieves: established normal account, then behavioral discontinuity
# ---------------------------------------------------------------------------
def gen_thief(rng, world, cfg, e=0.0):
    takeover = int(cfg.start_ts + rng.uniform(0.30, 0.85) * cfg.days * DAY)
    user, ev = _normal_core(rng, world, cfg, sub="regular", hard_end=takeover)
    home = user["_home"]
    topic_ids, weights = user["_topics"]
    vic_tok = user["_tok"]
    vic_fail = user["_fail"]
    _strip(user)

    # attacker phase: new device fingerprint is unavoidable (robust signal);
    # evasive attackers use a same-country residential proxy, mimic the
    # victim's prompt style, and ramp up volume gradually instead of bursting
    atk_dev = world.device_fp()
    atk_net = (world.residential(country=home[3]) if rng.random() < 0.7 * e
               else world.datacenter(exclude_country=home[3]))
    atk_ua = _pick(rng, UA_DESKTOP) if rng.random() < e else _pick(rng, UA_BOT)
    mimic_p = 0.5 * e
    vic_tok_mu, vic_tok_sigma = vic_tok
    atk_fail = (1 - e) * 0.10 + e * vic_fail
    t = takeover + rng.uniform(60, 3600)
    burst_end = min(cfg.end_ts, takeover + rng.uniform(1, 3 + 4 * e) * DAY)
    span = max(burst_end - takeover, 1)
    while t < burst_end:
        for _ in range(int(rng.integers(30, 120) * (1 - 0.7 * e) + 4)):
            if rng.random() < mimic_p:
                cluster, text = prompts.normal_prompt(rng, topic_ids, weights)
                tokens = rng.lognormal(vic_tok_mu, vic_tok_sigma)
            else:
                cluster, text = prompts.extract_prompt(rng)
                # evasive thieves keep request sizes near the victim's norm
                tokens = rng.lognormal((1 - e) * np.log(1500) + e * vic_tok_mu,
                                       0.5 + 0.3 * e)
            ev.add(t, atk_net, atk_dev, atk_ua, "/v1/chat", cluster, text,
                   tokens, rng.random() > atk_fail)
            # gradual ramp: gaps shrink from ~60s to ~8s across the burst
            progress = min(1.0, (t - takeover) / span)
            gap_mu = np.log(8) + e * (1 - progress) * (np.log(60) - np.log(8))
            t += max(1.0, rng.lognormal(gap_mu, 0.5))
            if t >= burst_end:
                break
        t += rng.uniform(2, 5 + 6 * e) * 3600  # pause between sessions

    user.update({"label": "token_thief", "subtype": "ato",
                 "ring_id": None, "takeover_ts": takeover, "evasion": e})
    return user, ev
