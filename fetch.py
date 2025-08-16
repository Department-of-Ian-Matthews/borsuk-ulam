import time
import requests

DEFAULT_TIMEOUT = 30
RETRIES = 3

def fetch_openmeteo_pair(lat: float, lon: float, alat: float, alon: float, sleep_between: float = 0.1):
    """Fetch current temp (C) & pressure (hPa) for (lat,lon) and antipode in one request."""
    lats = f"{lat:.4f},{alat:.4f}"
    lons = f"{lon:.4f},{alon:.4f}"
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}&current=temperature_2m,pressure_msl"
        "&timeformat=unixtime&forecast_days=1"
    )
    delay = sleep_between
    for _ in range(RETRIES):
        try:
            resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) == 2:
                    return {
                        "p1": {
                            "lat": data[0]["latitude"],
                            "lon": data[0]["longitude"],
                            "tempC": data[0]["current"]["temperature_2m"],
                            "pres": data[0]["current"]["pressure_msl"],
                        },
                        "p2": {
                            "lat": data[1]["latitude"],
                            "lon": data[1]["longitude"],
                            "tempC": data[1]["current"]["temperature_2m"],
                            "pres": data[1]["current"]["pressure_msl"],
                        },
                    }
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    return None
