"""
geocode.py — reverse geocoding helpers for the app

Notes:
- Nominatim requires a valid, identifying User-Agent (ideally with a contact email/URL).
- Be polite: keep >= 1 request/second (we use 1.1s) and cache results aggressively.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


# -------------------------------
# Configuration
# -------------------------------

# Provide a real contact so Nominatim can reach you if needed.
# You can set USER_AGENT via env var to avoid committing email to repo.
USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "borsuk-ulam/1.0 (youremail@example.com)"  # <-- REPLACE with a real contact
)

# Request timeout for each call (seconds)
TIMEOUT = float(os.getenv("NOMINATIM_TIMEOUT", "10"))

# Rate limiting and retries
MIN_DELAY_SECONDS = float(os.getenv("NOMINATIM_MIN_DELAY", "1.1"))  # >= 1 req/sec
MAX_RETRIES = int(os.getenv("NOMINATIM_MAX_RETRIES", "3"))

# MUST be >= MIN_DELAY_SECONDS to satisfy geopy's assertion.
ERROR_WAIT_SECONDS = float(os.getenv("NOMINATIM_ERROR_WAIT", "2.0"))

# Default zoom: 3 (country) … 18 (house). 10 ~ city/region
DEFAULT_ZOOM = int(os.getenv("NOMINATIM_DEFAULT_ZOOM", "10"))


# -------------------------------
# Geocoder + RateLimiter
# -------------------------------

_locator = Nominatim(user_agent=USER_AGENT, timeout=TIMEOUT)

_rate_limited_reverse = RateLimiter(
    _locator.reverse,
    min_delay_seconds=MIN_DELAY_SECONDS,
    max_retries=MAX_RETRIES,
    error_wait_seconds=ERROR_WAIT_SECONDS,  # MUST be >= min_delay_seconds
    swallow_exceptions=True,                # return None instead of raising
)


# -------------------------------
# Public helpers
# -------------------------------

@lru_cache(maxsize=50_000)
def closest_place(
    lat: float,
    lon: float,
    zoom: int = DEFAULT_ZOOM,
    include_country: bool = False,
) -> Optional[str]:
    """
    Reverse geocode (lat, lon) into a concise, human-friendly place name.

    Returns:
        str like "Chicago" or "Cook County" or "Illinois" (optionally with country),
        or None if not resolved / timed out / unavailable.
    """
    try:
        loc = _rate_limited_reverse(
            (lat, lon),
            language="en",
            addressdetails=True,
            zoom=zoom,
        )
        if not loc:
            return None

        addr: Dict[str, Any] = loc.raw.get("address", {})
        # Prefer city-like labels first, then fall back up the hierarchy.
        name = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or addr.get("county")
            or addr.get("state_district")
            or addr.get("state")
            or addr.get("region")
            or addr.get("country")
            or None
        )

        if not name:
            # As a last resort, return the full formatted address string.
            name = getattr(loc, "address", None)

        if not name:
            return None

        if include_country:
            country = addr.get("country")
            if country and country not in name:
                name = f"{name}, {country}"

        return name

    except (GeocoderTimedOut, GeocoderUnavailable):
        return None


@lru_cache(maxsize=50_000)
def reverse_details(
    lat: float,
    lon: float,
    zoom: int = DEFAULT_ZOOM,
) -> Optional[Dict[str, Any]]:
    """
    Reverse geocode and return the raw address dict (useful for debugging/UI).
    Returns None on failure.
    """
    try:
        loc = _rate_limited_reverse(
            (lat, lon),
            language="en",
            addressdetails=True,
            zoom=zoom,
        )
        if not loc:
            return None

        return {
            "address": loc.raw.get("address", {}),
            "display_name": getattr(loc, "address", None),
            "lat": getattr(loc, "latitude", None),
            "lon": getattr(loc, "longitude", None),
        }
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None


def place_label_or_placeholder(
    lat: float,
    lon: float,
    zoom: int = DEFAULT_ZOOM,
    placeholder: str = "(unresolved)",
) -> str:
    """
    Wrapper that never returns None—good for UI usage.
    """
    label = closest_place(lat, lon, zoom=zoom)
    return label or placeholder


# -------------------------------
# (Optional) Small utility
# -------------------------------

def normalize_latlon(lat: float, lon: float) -> Tuple[float, float]:
    """
    Clamp latitude to [-90, 90] and wrap longitude to [-180, 180].
    """
    clamped_lat = max(-90.0, min(90.0, float(lat)))
    lon = float(lon)
    wrapped_lon = ((lon + 180.0) % 360.0) - 180.0
    return clamped_lat, wrapped_lon
