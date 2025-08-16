import random
import time
import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import antipode, rel_diff, round_to_granularity
from db import connect, init_db, upsert_pair, get_matches, get_stats, next_unchecked_coords, exists_either
from fetch import fetch_openmeteo_pair
from geocode import closest_place

st.set_page_config(page_title="Borsuk-Ulam Globe", layout="wide")

st.title("🌍 Borsuk-Ulam Globe — Same temperature & pressure antipodes")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    db_path = st.text_input("SQLite DB path", value="borsuk_ulam.sqlite", help="File to persist results for resuming later.")
    granularity = st.number_input("Grid granularity (degrees)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Distance between grid points.")
    lat_min, lat_max = st.slider("Latitude range (°)", min_value=-90, max_value=90, value=(0, 90), step=1)
    lon_min, lon_max = st.slider("Longitude range (°)", min_value=-180, max_value=180, value=(-180, 180), step=1)

    st.divider()
    st.caption("Matching tolerances (relative %)")
    tol_temp = st.number_input("Temperature tolerance %", min_value=0.1, max_value=20.0, value=5.0, step=0.1) / 100.0
    tol_pres = st.number_input("Pressure tolerance %", min_value=0.1, max_value=20.0, value=5.0, step=0.1) / 100.0

    st.divider()
    st.caption("Rate limiting")
    calls_per_min = st.number_input("Max calls per minute", min_value=60, max_value=600, value=550, step=10)
    sleep_between = 60.0 / calls_per_min

    batch_to_seek = st.number_input("How many new coords to test when you click CONTINUE", min_value=1, max_value=2000, value=100, step=1)

    st.divider()
    do_reverse_geocode = st.checkbox("Annotate with nearest city (Nominatim, 1 req/sec)", value=True)

# --- DB init ---
con = connect(db_path)
init_db(con)

# --- Candidate grid ---
def build_candidates(lat_min, lat_max, lon_min, lon_max, gran):
    lats = np.arange(lat_min, lat_max + 1e-9, gran).tolist()
    lons = np.arange(lon_min, lon_max - 1e-9, gran).tolist()
    random.shuffle(lats)
    random.shuffle(lons)
    cand = []
    for la in lats:
        for lo in lons:
            la = float(round_to_granularity(la, gran))
            lo = float(round_to_granularity(lo, gran))
            ala, alo = antipode(la, lo)
            cand.append((la, lo, float(ala), float(alo)))
    random.shuffle(cand)
    return cand

candidates = build_candidates(lat_min, lat_max, lon_min, lon_max, granularity)

# --- Stats ---
stats = get_stats(con)
c1, c2 = st.columns(2)
with c1:
    st.metric("Total coords checked", stats["total_checked"])
with c2:
    st.metric("Total matches found", stats["matches"])

# --- Buttons ---
colA, colB = st.columns([1,1])
with colA:
    do_continue = st.button("▶️ CONTINUE", use_container_width=True)
with colB:
    clear_cache = st.button("🧹 Clear DB (danger)", use_container_width=True)

if clear_cache:
    st.warning("Clearing the current database...")
    con.close()
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    con = connect(db_path)
    init_db(con)
    st.success("DB cleared.")

# --- Core worker ---
def test_one_pair(lat, lon, tol_temp, tol_pres):
    a_lat, a_lon = antipode(lat, lon)
    # If either side exists, skip (already tested)
    if exists_either(con, lat, lon, a_lat, a_lon):
        return None

    data = fetch_openmeteo_pair(lat, lon, a_lat, a_lon, sleep_between=sleep_between)
    if not data:
        # Insert a placeholder record (no values), so we don't hammer the API repeatedly
        now = dt.datetime.utcnow().isoformat()
        upsert_pair(con, {
            "lat": lat, "lon": lon, "alat": a_lat, "alon": a_lon,
            "tempK": None, "pres": None, "atempK": None, "apres": None,
            "temp_rel_diff": None, "pres_rel_diff": None, "is_match": 0, "checked_at": now
        })
        return None

    t1C = data["p1"]["tempC"]; p1 = data["p1"]["pres"]
    t2C = data["p2"]["tempC"]; p2 = data["p2"]["pres"]
    # Convert to Kelvin to match your original code
    t1K = t1C + 273.15
    t2K = t2C + 273.15

    rtemp = rel_diff(t1K, t2K)
    rpres = rel_diff(p1, p2)
    match = int((rtemp <= tol_temp) and (rpres <= tol_pres))

    now = dt.datetime.utcnow().isoformat()
    upsert_pair(con, {
        "lat": lat, "lon": lon, "alat": a_lat, "alon": a_lon,
        "tempK": t1K, "pres": p1, "atempK": t2K, "apres": p2,
        "temp_rel_diff": rtemp, "pres_rel_diff": rpres, "is_match": match, "checked_at": now
    })
    return (lat, lon, a_lat, a_lon, t1K, p1, t2K, p2, rtemp, rpres, match)

# --- When continue is pressed ---
progress = st.empty()
log = st.empty()
if do_continue:
    to_try = next_unchecked_coords(con, candidates, int(batch_to_seek))
    n = len(to_try)
    if n == 0:
        st.info("No unchecked coordinates left in the current grid/range. Adjust the sliders or granularity.")
    else:
        for i, (la, lo, ala, alo) in enumerate(to_try, start=1):
            progress.progress(i / n, text=f"Checking {i}/{n}: ({la:.2f}, {lo:.2f}) & antipode")
            res = test_one_pair(la, lo, tol_temp, tol_pres)
            if res and res[-1] == 1:
                log.write(f"✅ Match: ({la:.2f}, {lo:.2f}) ↔ ({ala:.2f}, {alo:.2f})")
            time.sleep(sleep_between)
        progress.empty()
        st.success("Done testing this batch.")

# --- Load matches for plotting ---
rows = get_matches(con)
df = pd.DataFrame(rows, columns=["lat", "lon", "alat", "alon", "tempK", "pres", "atempK", "apres"])

# Optional reverse geocoding (slow-ish: ~1 req/sec)
if do_reverse_geocode and len(df) > 0:
    names = []
    dists = []
    anames = []
    adists = []
    for _, r in df.iterrows():
        name, dist = closest_place(r["lat"], r["lon"])
        aname, adist = closest_place(r["alat"], r["alon"])
        names.append(name); dists.append(dist)
        anames.append(aname); adists.append(adist)
    df["place"] = names
    df["place_km"] = dists
    df["aplace"] = anames
    df["aplace_km"] = adists

# --- 3D globe plot with Plotly ---
def make_globe_fig(df: pd.DataFrame):
    # Create a sphere mesh
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.cos(v))  # corrected order to form sphere
    zs = np.outer(np.ones_like(u), np.cos(v))

    sphere = go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.15,
        showscale=False
    )

    def latlon_to_xyz(lat, lon):
        latr = np.radians(lat)
        lonr = np.radians(lon)
        x = np.cos(latr) * np.cos(lonr)
        y = np.cos(latr) * np.sin(lonr)
        z = np.sin(latr)
        return x, y, z

    if len(df) == 0:
        scatter = go.Scatter3d()
        ascatter = go.Scatter3d()
    else:
        xs_p, ys_p, zs_p = latlon_to_xyz(df["lat"].values, df["lon"].values)
        xs_a, ys_a, zs_a = latlon_to_xyz(df["alat"].values, df["alon"].values)

        hovertext_p = []
        hovertext_a = []
        for i, r in df.iterrows():
            # Build hover strings
            t1 = f"Point: ({r['lat']:.2f}, {r['lon']:.2f})<br>T={r['tempK']:.2f} K, P={r['pres']:.1f} hPa"
            t2 = f"Antipode: ({r['alat']:.2f}, {r['alon']:.2f})<br>T={r['atempK']:.2f} K, P={r['apres']:.1f} hPa"
            if 'place' in df.columns:
                t1 += f"<br>Near: {r['place']} (~{r['place_km']:.1f} km)"
            if 'aplace' in df.columns:
                t2 += f"<br>Near: {r['aplace']} (~{r['aplace_km']:.1f} km)"
            hovertext_p.append(t1)
            hovertext_a.append(t2)

        scatter = go.Scatter3d(
            x=xs_p, y=ys_p, z=zs_p,
            mode="markers",
            marker=dict(size=4),
            name="Matches",
            hovertext=hovertext_p,
            hoverinfo="text"
        )
        ascatter = go.Scatter3d(
            x=xs_a, y=ys_a, z=zs_a,
            mode="markers",
            marker=dict(size=4, symbol="x"),
            name="Antipodes",
            hovertext=hovertext_a,
            hoverinfo="text"
        )

    fig = go.Figure(data=[sphere, scatter, ascatter])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    return fig

st.subheader("3D matches on unit sphere")
fig = make_globe_fig(df)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.subheader("Matched pairs")
st.dataframe(df)

st.info(
    "Tip: Put the SQLite file in a shared folder (e.g., network drive or cloud sync). "
    "Anyone running this app and pointing to the same DB path can continue where you left off."
)
