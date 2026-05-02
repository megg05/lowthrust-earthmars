import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def load_states(path):
    rows = []
    pattern = re.compile(
        r'^\s*([0-9.]+),\s*([^,]+?),\s*'
        r'([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+|[+-]?\d+(?:\.\d+)?)\s*,\s*'
        r'([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+|[+-]?\d+(?:\.\d+)?)\s*,\s*'
        r'([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+|[+-]?\d+(?:\.\d+)?)\s*,\s*'
        r'([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+|[+-]?\d+(?:\.\d+)?)\s*,\s*'
        r'([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+|[+-]?\d+(?:\.\d+)?)\s*,\s*'
        r'([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+|[+-]?\d+(?:\.\d+)?)'
    )

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                jd = float(m.group(1))
                dt_str = m.group(2).strip()
                x, y, z, vx, vy, vz = map(float, m.groups()[2:])
                rows.append([jd, dt_str, x, y, z, vx, vy, vz])

    df = pd.DataFrame(rows, columns=["jd", "datetime", "x", "y", "z", "vx", "vy", "vz"])

    state_vectors = df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    def parse_horizons_dt(s):
        s = s.replace("A.D. ", "").strip()
        return datetime.strptime(s, "%Y-%b-%d %H:%M:%S.%f")

    datetimes = df["datetime"].apply(parse_horizons_dt)

    return state_vectors, datetimes

def query_states(df):
    pass


