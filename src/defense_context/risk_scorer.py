"""
risk_scorer.py
==============
Defense-Context Risk Scoring Module

WHAT THIS DOES:
  Takes the CNN's stego probability + operational metadata about the
  image transfer, and produces a COMPOSITE RISK SCORE.

  This is Innovation 2 of the project. The idea is:
    "Don't just say 'this image has hidden data' — say HOW DANGEROUS it is."

  A stego image sent by an admin at 2pm on a work laptop is LOW risk.
  The same image sent by an intern at 3am to an external IP is CRITICAL.

RISK FACTORS:
  1. ML Confidence: how sure is the CNN that this is stego?
  2. User Privilege: admin vs normal vs intern/guest
  3. Time of Transfer: business hours vs late night vs weekend
  4. File Size Anomaly: is the file much larger than expected?
  5. Network Destination: internal network vs external IP

SAMPLE OUTPUT:
  "Image is 94.7% likely STEGO. Sent by intern account at 2:18 AM
   to external IP 103.x.x.x. Risk Level: CRITICAL"

STUDENT NOTE:
  This module is rule-based (no ML needed). It shows practical
  engineering thinking — a security officer needs ACTIONABLE
  intelligence, not just a probability number.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, time as dtime
import ipaddress


# ─── Risk Levels ─────────────────────────────────────────────────────────────

RISK_LEVELS = {
    0: "CLEAN",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "CRITICAL"
}

RISK_COLORS = {
    "CLEAN":    "#27AE60",   # green
    "LOW":      "#2ECC71",   # light green
    "MEDIUM":   "#F39C12",   # orange
    "HIGH":     "#E67E22",   # dark orange
    "CRITICAL": "#E74C3C"    # red
}

# ─── Input Data Classes ───────────────────────────────────────────────────────

@dataclass
class TransferMetadata:
    """
    Metadata about the image transfer event.
    All fields are optional — unset fields contribute 0 to risk score.
    """
    # Who sent the file
    user_role: str = "normal"          # "admin", "normal", "intern", "guest", "unknown"

    # When was it sent (as a datetime object, or None)
    timestamp: Optional[datetime] = None

    # File properties
    file_size_bytes: Optional[int]   = None    # actual file size
    expected_size_bytes: Optional[int] = None  # expected size for this resolution

    # Network
    destination_ip: Optional[str] = None       # where the file is going
    source_ip:      Optional[str] = None       # where the file came from

    # Extra flags
    is_encrypted_channel: bool = False          # if True, slight risk increase
    is_repeat_offender:   bool = False          # user previously flagged?


@dataclass
class RiskResult:
    """Output of the risk scoring process."""
    risk_score:      int     # 0-100
    risk_level:      str     # CLEAN / LOW / MEDIUM / HIGH / CRITICAL
    risk_color:      str     # hex color for UI
    stego_prob:      float   # CNN probability (0.0-1.0)
    contributing_factors: list = field(default_factory=list)  # list of strings
    alert_message:   str     = ""

    def __str__(self) -> str:
        return (
            f"Risk Level: {self.risk_level} (score: {self.risk_score}/100)\n"
            f"Stego probability: {self.stego_prob*100:.1f}%\n"
            f"Factors: {', '.join(self.contributing_factors)}\n"
            f"Alert: {self.alert_message}"
        )


# ─── Scoring Functions ────────────────────────────────────────────────────────

def _score_cnn_confidence(stego_prob: float) -> tuple[int, str]:
    """Score based on CNN's stego probability."""
    if stego_prob < 0.3:
        return 0, None
    elif stego_prob < 0.5:
        return 10, f"Low stego signal ({stego_prob*100:.1f}% confidence)"
    elif stego_prob < 0.7:
        return 25, f"Moderate stego signal ({stego_prob*100:.1f}% confidence)"
    elif stego_prob < 0.85:
        return 40, f"Strong stego signal ({stego_prob*100:.1f}% confidence)"
    else:
        return 55, f"Very high stego confidence ({stego_prob*100:.1f}%)"


def _score_user_privilege(user_role: str) -> tuple[int, str]:
    """Higher risk for lower-privilege users (interns shouldn't handle classified images)."""
    role_scores = {
        "admin":   0,     # trusted
        "normal":  5,     # standard user — slight risk
        "intern":  15,    # interns shouldn't transfer sensitive images
        "guest":   20,    # guests have no business with classified images
        "unknown": 25,    # unknown user = biggest privilege concern
    }
    score = role_scores.get(user_role.lower(), 10)
    if score == 0:
        return 0, None
    return score, f"User role: {user_role} (low privilege)"


def _score_transfer_time(timestamp: Optional[datetime]) -> tuple[int, str]:
    """Off-hours transfers are suspicious."""
    if timestamp is None:
        return 0, None

    hour = timestamp.hour
    is_weekend = timestamp.weekday() >= 5   # Sat=5, Sun=6

    if is_weekend:
        return 15, f"Weekend transfer at {timestamp.strftime('%A %H:%M')}"

    if 0 <= hour < 6:
        return 20, f"Late night transfer at {timestamp.strftime('%H:%M')} (high risk)"
    elif 6 <= hour < 9 or 20 <= hour < 24:
        return 10, f"Off-hours transfer at {timestamp.strftime('%H:%M')}"
    else:
        return 0, None   # business hours = normal


def _score_file_size_anomaly(
    actual_bytes: Optional[int],
    expected_bytes: Optional[int]
) -> tuple[int, str]:
    """
    Steganographic images can be larger than expected for their resolution.
    A 5MB PNG where 500KB is typical = suspicious.
    """
    if actual_bytes is None or expected_bytes is None:
        return 0, None

    if expected_bytes == 0:
        return 0, None

    ratio = actual_bytes / expected_bytes

    if ratio < 1.5:
        return 0, None
    elif ratio < 3.0:
        return 5, f"File size {ratio:.1f}× larger than expected"
    elif ratio < 6.0:
        return 10, f"File size {ratio:.1f}× larger than expected (suspicious)"
    else:
        return 15, f"File size {ratio:.1f}× larger than expected (very suspicious)"


def _score_network_destination(destination_ip: Optional[str]) -> tuple[int, str]:
    """
    External IPs are more dangerous than internal network transfers.
    """
    if destination_ip is None:
        return 0, None

    try:
        ip = ipaddress.ip_address(destination_ip)

        # Private IP ranges = internal network = lower risk
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return 2, None   # internal — minimal risk

        # Public IP = external destination = HIGH risk
        # (Data is leaving the organization)
        return 20, f"External destination IP: {destination_ip} (data leaving network)"

    except ValueError:
        # Couldn't parse IP — flag as unknown
        return 5, f"Unknown destination: {destination_ip}"


def _score_repeat_offender(is_repeat: bool) -> tuple[int, str]:
    if not is_repeat:
        return 0, None
    return 15, "User previously flagged for suspicious transfers"


# ─── Main Scoring Function ────────────────────────────────────────────────────

def compute_risk_score(
    stego_prob: float,
    metadata: Optional[TransferMetadata] = None
) -> RiskResult:
    """
    Compute composite risk score combining CNN output + operational metadata.

    Args:
        stego_prob: CNN output probability of stego (0.0 - 1.0)
        metadata  : optional transfer metadata (all factors are optional)

    Returns:
        RiskResult with score, level, alert message
    """
    if metadata is None:
        metadata = TransferMetadata()

    total_score = 0
    factors = []

    # Factor 1: CNN confidence (most important)
    s, msg = _score_cnn_confidence(stego_prob)
    total_score += s
    if msg:
        factors.append(msg)

    # Factor 2: User privilege
    s, msg = _score_user_privilege(metadata.user_role)
    total_score += s
    if msg:
        factors.append(msg)

    # Factor 3: Transfer time
    s, msg = _score_transfer_time(metadata.timestamp)
    total_score += s
    if msg:
        factors.append(msg)

    # Factor 4: File size anomaly
    s, msg = _score_file_size_anomaly(
        metadata.file_size_bytes,
        metadata.expected_size_bytes
    )
    total_score += s
    if msg:
        factors.append(msg)

    # Factor 5: Network destination
    s, msg = _score_network_destination(metadata.destination_ip)
    total_score += s
    if msg:
        factors.append(msg)

    # Factor 6: Repeat offender
    s, msg = _score_repeat_offender(metadata.is_repeat_offender)
    total_score += s
    if msg:
        factors.append(msg)

    # Clamp to 0-100
    total_score = min(100, max(0, total_score))

    # Map score to level
    if total_score == 0:
        level = "CLEAN"
    elif total_score < 20:
        level = "LOW"
    elif total_score < 45:
        level = "MEDIUM"
    elif total_score < 70:
        level = "HIGH"
    else:
        level = "CRITICAL"

    # Build alert message
    if stego_prob < 0.5 and total_score < 20:
        alert = "Image appears clean. No action required."
    elif stego_prob >= 0.5 and total_score >= 70:
        alert = (
            f"Image is {stego_prob*100:.1f}% likely STEGO. "
            f"Sent by {metadata.user_role} account"
            + (f" at {metadata.timestamp.strftime('%H:%M')}" if metadata.timestamp else "")
            + (f" to external IP {metadata.destination_ip}" if metadata.destination_ip else "")
            + f". Risk Level: {level}. Immediate review recommended."
        )
    elif stego_prob >= 0.5:
        alert = (
            f"Image is {stego_prob*100:.1f}% likely STEGO. "
            f"Risk Level: {level}. Flag for security review."
        )
    else:
        alert = (
            f"Image has low stego probability ({stego_prob*100:.1f}%) "
            f"but elevated context risk. Monitor transfer."
        )

    return RiskResult(
        risk_score=total_score,
        risk_level=level,
        risk_color=RISK_COLORS.get(level, "#95A5A6"),
        stego_prob=stego_prob,
        contributing_factors=factors,
        alert_message=alert
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Defense-Context Risk Scorer — Demo\n")

    # Scenario 1: High-confidence stego from intern at 2 AM to external IP
    meta1 = TransferMetadata(
        user_role="intern",
        timestamp=datetime(2026, 8, 8, 2, 18),    # 2:18 AM
        file_size_bytes=5_200_000,                 # 5.2 MB
        expected_size_bytes=500_000,               # 500 KB expected
        destination_ip="103.45.67.89"              # external IP
    )
    result1 = compute_risk_score(stego_prob=0.947, metadata=meta1)
    print("Scenario 1: Intern at 2 AM to external IP")
    print(result1)
    print()

    # Scenario 2: Low confidence, admin user, business hours
    meta2 = TransferMetadata(
        user_role="admin",
        timestamp=datetime(2026, 8, 8, 14, 30),   # 2:30 PM
        destination_ip="192.168.1.50"              # internal IP
    )
    result2 = compute_risk_score(stego_prob=0.35, metadata=meta2)
    print("Scenario 2: Admin at 2:30 PM, internal network")
    print(result2)
    print()

    # Scenario 3: No metadata (CNN-only scoring)
    result3 = compute_risk_score(stego_prob=0.82)
    print("Scenario 3: No metadata provided (CNN confidence only)")
    print(result3)
