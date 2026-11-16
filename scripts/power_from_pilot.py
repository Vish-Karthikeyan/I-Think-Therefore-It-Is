"""Power for the main run from each model's pilot inputs (Section 4.7). Sizes are trials per
identity group per cue type in Tier 2 (contradicted; neutral = about half). Effects: Delta_b
at the pilot estimate; theta_cong at three candidate sizes. No API calls."""
import sys
import warnings

import pandas as pd

from idea.analysis import power, rq1
from idea.runner import derive
from idea.stance import attach_g

warnings.filterwarnings("ignore")
rd = sys.argv[1]
rows = []
for m in ("gpt", "claude"):
    df = derive(pd.read_parquet(f"{rd}/responses/{m}_chat.parquet"))
    st = pd.read_parquet(f"{rd}/stance/{m}_chat.parquet"); st["fmt"] = "chat"
    df = attach_g(df, st)
    inp = power.pilot_inputs(df, m)
    db = float(rq1.primary(rq1.analysis_frame(df, 2, m))["delta_b"].est)
    for theta in (-0.05, -0.10, -0.15):
        for n in (75, 150, 225, 283):
            res = power.power(reps=100, n_contra=2 * n, n_neutral=n, delta_b=db, theta=theta, **inp)
            rows.append(dict(model=m, theta=theta, delta_b=round(db, 3), n_contra_per_group_cue=n, **res))
            print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"{rd}/power_main_run.csv", index=False)
print(out.to_string(index=False))
