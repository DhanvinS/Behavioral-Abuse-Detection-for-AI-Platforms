"""Network/device world model: IP pools, ASNs, device fingerprints, user agents.

Networks are returned as tuples: (ip, asn, asn_type, country).
"""
import numpy as np

# (asn, country) pools per network type
RES_ASNS = [(7922, "US"), (701, "US"), (20115, "US"), (22773, "US"),
            (3320, "DE"), (2856, "GB"), (4766, "KR"), (45609, "IN"), (8151, "MX")]
MOBILE_ASNS = [(21928, "US"), (20057, "US"), (310, "US"), (38266, "IN"), (23693, "ID")]
DC_ASNS = [(16509, "US"), (14061, "US"), (24940, "DE"), (16276, "FR"),
           (9009, "RO"), (45102, "SG"), (200019, "RU"), (135377, "HK")]
CORP_ASNS = [(36351, "US"), (8075, "US"), (15169, "US")]
CAMPUS_ASN = (27, "US")  # AS27: University of Maryland

UA_DESKTOP = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/124.0.2478.97",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) Chrome/124.0.0.0",
]
UA_MOBILE = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) Chrome/125.0.0.0 Mobile",
    "Mozilla/5.0 (Linux; Android 13; SM-S918B) Chrome/124.0.0.0 Mobile",
]
UA_BOT = [
    "python-requests/2.31.0",
    "python-httpx/0.27.0",
    "Go-http-client/2.0",
    "axios/1.6.8",
    "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/120.0.6099.109",
]

_USERNAME_WORDS = [
    "swift", "lunar", "pixel", "cobalt", "ember", "quartz", "nova", "drift",
    "maple", "cedar", "raven", "atlas", "echo", "zephyr", "orbit", "sage",
    "blaze", "frost", "comet", "willow", "onyx", "delta", "rune", "vivid",
]


class World:
    """Allocates IPs, devices, and identities. Holds shared NAT pools."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # shared NAT egress IPs: many legit users behind each (hard negatives)
        self.campus_ips = [self._rand_ip(128, 8) for _ in range(6)]
        self.corporate_nats = [(self._rand_ip(), *CORP_ASNS[i % len(CORP_ASNS)])
                               for i in range(8)]

    def _rand_ip(self, a=None, b=None) -> str:
        r = self.rng
        a = a if a is not None else int(r.integers(11, 223))
        b = b if b is not None else int(r.integers(0, 256))
        return f"{a}.{b}.{int(r.integers(0, 256))}.{int(r.integers(1, 255))}"

    def _pick(self, pool):
        return pool[int(self.rng.integers(len(pool)))]

    # ---- network allocators -------------------------------------------------
    def residential(self, country=None):
        opts = [p for p in RES_ASNS if p[1] == country] if country else RES_ASNS
        asn, cc = self._pick(opts or RES_ASNS)
        return (self._rand_ip(), asn, "residential", cc)

    def mobile_carrier(self, country=None):
        """Returns (asn, cc); call mobile_ip(carrier) per session for CGNAT churn."""
        opts = [p for p in MOBILE_ASNS if p[1] == country] if country else MOBILE_ASNS
        return self._pick(opts or MOBILE_ASNS)

    def mobile_ip(self, carrier):
        asn, cc = carrier
        r = self.rng
        ip = f"100.{int(r.integers(64, 128))}.{int(r.integers(0, 256))}.{int(r.integers(1, 255))}"
        return (ip, asn, "mobile", cc)

    def datacenter(self, exclude_country=None):
        opts = [p for p in DC_ASNS if p[1] != exclude_country] if exclude_country else DC_ASNS
        asn, cc = self._pick(opts or DC_ASNS)
        return (self._rand_ip(), asn, "datacenter", cc)

    def campus(self):
        ip = self._pick(self.campus_ips)
        return (ip, CAMPUS_ASN[0], "university", CAMPUS_ASN[1])

    def corporate(self):
        ip, asn, cc = self._pick(self.corporate_nats)
        return (ip, asn, "corporate", cc)

    # ---- identities ----------------------------------------------------------
    def device_fp(self) -> str:
        return f"fp_{self.rng.integers(0, 1 << 48):012x}"

    def payment_hash(self) -> str:
        return f"pay_{self.rng.integers(0, 1 << 40):010x}"

    def username(self) -> str:
        w1, w2 = self._pick(_USERNAME_WORDS), self._pick(_USERNAME_WORDS)
        return f"{w1}_{w2}{int(self.rng.integers(1, 9999))}"

    def farm_username(self, base: str, idx: int) -> str:
        return f"{base}{idx:03d}"
