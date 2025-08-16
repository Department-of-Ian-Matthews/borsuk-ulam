import math
from typing import Tuple

EARTH_RADIUS_KM = 6371.0088

def antipode(lat: float, lon: float) -> Tuple[float, float]:
    """Return the antipode of (lat, lon)."""
    alat = -lat
    alon = lon - 180 if lon > 0 else lon + 180
    # Normalize longitude to [-180, 180)
    if alon >= 180:
        alon -= 360
    if alon < -180:
        alon += 360
    return (alat, alon)

def rel_diff(x: float, y: float) -> float:
    """Relative difference: |x - y| / mean(|x|, |y|)."""
    denom = (abs(x) + abs(y)) / 2.0
    if denom == 0:
        return 0.0
    return abs(x - y) / denom

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points (degrees)."""
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_KM * c

def round_to_granularity(x: float, gran: float) -> float:
    """Snap a floating-point value to a grid step (gran)."""
    return round(x / gran) * gran
