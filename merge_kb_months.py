#!/usr/bin/env python
#!/usr/bin/env python
"""
Merge monthly kb outputs (from kb_one_month.py) into one file with a `month` dimension.
Assumes files are named:
  kb_truth_monthMM.nc
  kb_ml_monthMM.nc
  kb_semi_monthMM.nc
  kb_u10_only_monthMM.nc
and located in OUTDIR below.
"""

import os
import glob
import numpy as np
import xarray as xr
import re

# === USER SETTINGS ===
OUTDIR = "/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/Analysis/NNWave/Postprocess/GasFlux_SeaSaltEmission/KB_MONTHLY_NO_DASK"
OUTNAME = "kb_monthly_2019_2022.nc"

# ======================

PAT = re.compile(r".*month(\d{2})\.nc$")

def _stack_one(pattern, varname):
    files = sorted(glob.glob(os.path.join(OUTDIR, pattern)))
    if not files:
        raise SystemExit(f"[{varname}] No files match {pattern} in {OUTDIR}")

    months, das = [], []
    for f in files:
        m = PAT.match(f)
        if not m:
            raise SystemExit(f"File name does not contain monthMM: {f}")
        mo = int(m.group(1))
        ds = xr.open_dataset(f, chunks=None)
        # use variable name directly (each file has one variable)
        da = list(ds.data_vars.values())[0]
        da = da.expand_dims(month=[mo])
        das.append(da)
        months.append(mo)

    order = np.argsort(months)
    stacked = xr.concat([das[i] for i in order], dim="month").sortby("month")
    return stacked.to_dataset(name=varname)

def main():
    print("Merging monthly kb files from:", OUTDIR)
    ds = xr.merge([
        _stack_one("kb_truth_month*.nc",    "kb_truth"),
        _stack_one("kb_ml_month*.nc",       "kb_ml"),
        _stack_one("kb_semi_month*.nc",     "kb_semi"),
        _stack_one("kb_u10_only_month*.nc", "kb_u10_only"),
    ])

    enc = {v: dict(zlib=True, complevel=4, dtype="float32") for v in ds.data_vars}
    outpath = os.path.join(OUTDIR, OUTNAME)
    tmp = outpath + ".tmp"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, outpath)
    print(f"\n✅ Wrote merged file:\n  {outpath}\n")

if __name__ == "__main__":
    main()
