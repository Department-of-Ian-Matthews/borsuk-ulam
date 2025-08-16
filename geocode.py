from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from utils import haversine_km

_locator = Nominatim(user_agent="borsuk-ulam-globe-app/1.0 (educational)")
_reverse = RateLimiter(_locator.reverse, min_delay_seconds=1.0)

def closest_place(lat: float, lon: float):
    """Reverse geocode (lat, lon) to name and distance (km)."""
    try:
        loc = _reverse((lat, lon), language="en", zoom=10, addressdetails=True, exactly_one=True)
        if loc is None:
            return ("Unknown", float("nan"))
        name = None
        if "address" in loc.raw:
            addr = loc.raw["address"]
            name = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("suburb")
        if not name:
            name = loc.raw.get("display_name", "Unknown").split(",")[0]
        dist = haversine_km(lat, lon, loc.latitude, loc.longitude)
        return (name, dist)
    except Exception:
        return ("Unknown", float("nan"))
