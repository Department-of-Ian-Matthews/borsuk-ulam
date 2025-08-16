import os
import time
import random
import sqlite3
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from utils import antipode, rel_diff, round_to_granularity
from db import (
    connect, init_db, upsert_pair, get_matches, get_stats,
    next_unchecked_coords, exists_either
)
from fetch import fetch_openmeteo_pair
from geocode import closest_place

# ---- Streamlit page ----
st.set_page_config(page_title="Borsuk-Ulam Globe", layout="wide")
st.title("🌍 Borsuk-Ulam Globe — Same temperature & pressure antipodes")

# ---- Make sure pydeck uses GlobeView and NOT Mapbox ----
os.environ.pop("MAPBOX_API_KEY", None)
os.environ.pop("PYDECK_MAPBOX_API_KEY", None)
try:
    pdk.settings.mapbox_api_key = None
except Exception:
    pass
# Pin deck.gl bundle that has GlobeView
pdk.settings.custom_libraries = [
    {"libraryName": "deck.gl", "resourceUri": "https://unpkg.com/deck.gl@8.9.36/dist.min.js"}
]

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    db_path = st.text_input(
        "SQLite DB path",
        value="borsuk_ulam.sqlite",
        help="File to persist results for resuming later."
    )
    granularity = st.number_input(
        "Grid granularity (degrees)", min_value=0.1, max_value=10.0,
        value=1.0, step=0.1, help="Distance between grid points."
    )
    lat_min, lat_max = st.slider(
        "Latitude range (°)", min_value=-90, max_value=90,
        value=(0, 90), step=1
    )
    lon_min, lon_max = st.slider(
        "Longitude range (°)", min_value=-180, max_value=180,
        value=(-180, 180), step=1
    )

    st.divider()
    st.caption("Matching tolerances (relative %)")
    tol_temp = st.number_input(
        "Temperature tolerance %", min_value=0.1, max_value=20.0,
        value=5.0, step=0.1
    ) / 100.0
    tol_pres = st.number_input(
        "Pressure tolerance %", min_value=0.1, max_value=20.0,
        value=5.0, step=0.1
    ) / 100.0

    st.divider()
    st.caption("Rate limiting")
    calls_per_min = st.number_input(
        "Max calls per minute", min_value=60, max_value=600,
        value=550, step=10
    )
    sleep_between = 60.0 / calls_per_min

    batch_to_seek = st.number_input(
        "How many new coords to test when you click CONTINUE",
        min_value=1, max_value=2000, value=100, step=1
    )
    
    st.divider()
    do_reverse_geocode = st.checkbox(
        "Annotate with nearest city (Nominatim, ~1 req/sec)",
        value=False
    )
    
    

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

# --- Stats (LIVE-UPDATING) ---
stats_cols = st.columns(2)
metric_checked = stats_cols[0].empty()
metric_matches = stats_cols[1].empty()

def refresh_metrics():
    s = get_stats(con)
    metric_checked.metric("Total coords checked", s["total_checked"])
    metric_matches.metric("Total matches found", s["matches"])

# initial render
refresh_metrics()

# --- Buttons ---
colA, colB = st.columns([1, 1])
with colA:
    do_continue = st.button("▶️ CONTINUE", use_container_width=True)
with colB:
    clear_cache = st.button("🧹 Clear DB (danger)", use_container_width=True)

# --- Clear DB (soft clear + VACUUM in autocommit) ---
if clear_cache:
    st.warning("Clearing the current database...")
    # Delete rows in a normal transaction
    with con:
        con.execute("DELETE FROM pairs;")
    # VACUUM must run outside a transaction; open a temp autocommit connection
    tmp = sqlite3.connect(db_path, isolation_level=None)
    try:
        tmp.execute("VACUUM")
    finally:
        tmp.close()
    refresh_metrics()
    st.success("DB cleared (rows deleted + compacted).")

# --- Core worker ---
def test_one_pair(lat, lon, tol_temp, tol_pres):
    a_lat, a_lon = antipode(lat, lon)
    # If either side exists, skip
    if exists_either(con, lat, lon, a_lat, a_lon):
        return None

    data = fetch_openmeteo_pair(lat, lon, a_lat, a_lon, sleep_between=sleep_between)
    if not data:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M")
        upsert_pair(con, {
            "lat": lat, "lon": lon, "alat": a_lat, "alon": a_lon,
            "tempK": None, "pres": None, "atempK": None, "apres": None,
            "temp_rel_diff": None, "pres_rel_diff": None,
            "is_match": 0, "checked_at": now
        })
        return None

    t1C = data["p1"]["tempC"]; p1 = data["p1"]["pres"]
    t2C = data["p2"]["tempC"]; p2 = data["p2"]["pres"]
    t1K = t1C + 273.15
    t2K = t2C + 273.15

    rtemp = rel_diff(t1K, t2K)
    rpres = rel_diff(p1, p2)
    match = int((rtemp <= tol_temp) and (rpres <= tol_pres))

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    upsert_pair(con, {
        "lat": lat, "lon": lon, "alat": a_lat, "alon": a_lon,
        "tempK": t1K, "pres": p1, "atempK": t2K, "apres": p2,
        "temp_rel_diff": rtemp, "pres_rel_diff": rpres,
        "is_match": match, "checked_at": now
    })
    return (lat, lon, a_lat, a_lon, t1K, p1, t2K, p2, rtemp, rpres, match, now)

# --- When CONTINUE is pressed ---
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
            if res and res[-2] == 1:
                log.write(f"✅ Match: ({la:.2f}, {lo:.2f}) ↔ ({ala:.2f}, {alo:.2f})")

            # LIVE metric refresh (every 5; adjust or set to 1 for every item)
            if i % 5 == 0 or i == n:
                refresh_metrics()

            time.sleep(sleep_between)

        progress.empty()
        refresh_metrics()  # final refresh
        st.success("Done testing this batch.")

# --- Load matches for plotting ---
rows = get_matches(con)
df = pd.DataFrame(
    rows,
    columns=["checked_at", "lat", "lon", "alat", "alon", "tempK", "pres", "atempK", "apres"]
)

# --- PYDECK GLOBE (render immediately) ---
st.subheader("3D matches on Earth (globe)")

# Small UI toggle
show_pair_lines = st.checkbox("Show great-circle lines between pairs", value=True)

if len(df) == 0:
    st.info("No matches yet.")
else:
    # Build plotting tables
    base = df[["lat", "lon", "tempK", "pres"]].copy()
    base["kind"] = "Match"
    base["pair_id"] = base.index  # stable id per row

    anti = df[["alat", "alon", "atempK", "apres"]].rename(
        columns={"alat": "lat", "alon": "lon", "atempK": "tempK", "apres": "pres"}
    )
    anti["kind"] = "Antipode"
    anti["pair_id"] = anti.index

    plot_df = pd.concat([base, anti], ignore_index=True)

    def color_for(kind):
        return [30, 144, 255] if kind == "Match" else [255, 140, 105]

    plot_df["color"] = plot_df["kind"].map(color_for)
    plot_df["radius"] = 60000  # meters; tweak if you change zoom

    # Data for arcs between each pair
    arc_df = df.assign(
        lat2=df["alat"], lon2=df["alon"]
    )[["lat", "lon", "lat2", "lon2"]].copy()

    # Globe + layers
    view = pdk.View("GlobeView")
    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=0.8)

    # Points (with a thin outline so hover is clearer)
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[lon, lat]",
        get_radius="radius",
        radius_min_pixels=2,
        radius_max_pixels=30,
        get_fill_color="color",
        get_line_color=[10, 10, 10, 120],   # subtle stroke
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    layers = [scatter]

    # Very light arcs connecting pairs (highlight when hovered)
    if show_pair_lines and len(arc_df) > 0:
        arcs = pdk.Layer(
            "GreatCircleLayer",         # uses globe great-circle geometry
            data=arc_df,
            get_source_position="[lon, lat]",
            get_target_position="[lon2, lat2]",
            get_width=1.5,
            get_source_color=[80, 160, 255, 60],   # faint
            get_target_color=[255, 120, 120, 60],  # faint
            pickable=True,
            auto_highlight=True,
        )
        layers.append(arcs)

    # Tooltip
    tooltip = {
        "html": (
            "<b>{kind}</b>"
            "<br/>Lat: {lat}"
            "<br/>Lon: {lon}"
            "<br/>T: {tempK} K"
            "<br/>P: {pres} hPa"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"},
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[view],
        tooltip=tooltip,
        map_provider=None,   # we are using GlobeView (no flat map)
        map_style=None,
    )
    st.pydeck_chart(deck, use_container_width=True)


# --- Optional reverse geocoding AFTER rendering (slow) ---
if do_reverse_geocode and len(df) > 0:
    with st.status("Annotating with nearest cities...", expanded=False) as status:
        names, dists, anames, adists = [], [], [], []
        for _, r in df.iterrows():
            try:
                name, dist = closest_place(r["lat"], r["lon"])
            except Exception:
                name, dist = "Unknown", float("nan")
            try:
                aname, adist = closest_place(r["alat"], r["alon"])
            except Exception:
                aname, adist = "Unknown", float("nan")
            names.append(name); dists.append(dist)
            anames.append(aname); adists.append(adist)
        df["place"] = names; df["place_km"] = dists
        df["aplace"] = anames; df["aplace_km"] = adists
        status.update(label="Done annotating.", state="complete")

st.subheader("Matched pairs")
st.dataframe(df)

st.info(
    "Tip: Put the SQLite file in a shared folder (e.g., network drive or cloud sync). "
    "Anyone running this app and pointing to the same DB path can continue where you left off."
)
