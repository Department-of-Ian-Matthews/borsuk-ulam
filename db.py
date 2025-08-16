import sqlite3
from typing import Any, Dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    alat REAL NOT NULL,
    alon REAL NOT NULL,
    tempK REAL,
    pres REAL,
    atempK REAL,
    apres REAL,
    temp_rel_diff REAL,
    pres_rel_diff REAL,
    is_match INTEGER NOT NULL DEFAULT 0,
    checked_at TEXT NOT NULL,
    UNIQUE(lat, lon)
);
CREATE INDEX IF NOT EXISTS idx_pairs_match ON pairs(is_match);
CREATE INDEX IF NOT EXISTS idx_pairs_checked_at ON pairs(checked_at);
"""

def connect(db_path: str):
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db(con):
    cur = con.cursor()
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            cur.execute(s)
    con.commit()

def upsert_pair(con, row: Dict[str, Any]):
    """Insert or replace a pair record by (lat, lon)."""
    con.execute(
        """
        INSERT INTO pairs(lat, lon, alat, alon, tempK, pres, atempK, apres, temp_rel_diff, pres_rel_diff, is_match, checked_at)
        VALUES(:lat, :lon, :alat, :alon, :tempK, :pres, :atempK, :apres, :temp_rel_diff, :pres_rel_diff, :is_match, :checked_at)
        ON CONFLICT(lat, lon) DO UPDATE SET
            alat=excluded.alat,
            alon=excluded.alon,
            tempK=excluded.tempK,
            pres=excluded.pres,
            atempK=excluded.atempK,
            apres=excluded.apres,
            temp_rel_diff=excluded.temp_rel_diff,
            pres_rel_diff=excluded.pres_rel_diff,
            is_match=excluded.is_match,
            checked_at=excluded.checked_at
        """,
        row,
    )
    con.commit()

def exists_either(con, lat: float, lon: float, alat: float, alon: float) -> bool:
    cur = con.cursor()
    cur.execute(
        "SELECT 1 FROM pairs WHERE (lat=? AND lon=?) OR (lat=? AND lon=?) LIMIT 1",
        (lat, lon, alat, alon),
    )
    return cur.fetchone() is not None

def get_matches(con) -> list:
    cur = con.cursor()
    cur.execute("SELECT lat, lon, alat, alon, tempK, pres, atempK, apres FROM pairs WHERE is_match=1")
    return cur.fetchall()

def get_stats(con) -> dict:
    cur = con.cursor()
    cur.execute("SELECT COUNT(*), SUM(is_match) FROM pairs")
    total, matches = cur.fetchone()
    matches = matches or 0
    return {"total_checked": total, "matches": matches}

def next_unchecked_coords(con, candidates: list, limit: int) -> list:
    """Return up to `limit` coords from `candidates` that are not in DB (nor their antipodes)."""
    out = []
    cur = con.cursor()
    for (lat, lon, alat, alon) in candidates:
        cur.execute(
            "SELECT 1 FROM pairs WHERE (lat=? AND lon=?) OR (lat=? AND lon=?) LIMIT 1",
            (lat, lon, alat, alon),
        )
        if cur.fetchone() is None:
            out.append((lat, lon, alat, alon))
        if len(out) >= limit:
            break
    return out
