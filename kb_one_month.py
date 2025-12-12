#!/usr/bin/env python
import os
import argparse
import numpy as np
import xarray as xr

# ------------------ Thermophysical helpers (expect T in °C) ------------------

def solco2_all(T):
    r1 = 8.314
    a  = [-60.2409, 93.4157, 23.3585]
    b  = [0.023517, -0.023656, 0.0047036]
    a0 = [2073.1, 125.632, 3.6276, 0.043219]
    tsx = (T + 273.16) / 100.0
    s   = a[0] + a[1]/tsx + a[2]*np.log(tsx) + 35.0*(b[0] + b[1]*tsx + b[2]*tsx*tsx)
    sol = np.exp(s) * 1000.0
    al  = (sol * tsx) / 1e5 * r1 * 100.0
    sc  = a0[0] - a0[1]*T + a0[2]*T**2 - a0[3]*T**3
    return sol, al, sc

def solco2_alpha(T): return solco2_all(T)[1]
def solco2_Sc(T):    return solco2_all(T)[2]

def SW_Psat2(T, S):
    # T in °C
    TK = T + 273.15
    a = [-5.8002206E+03, 1.3914993E+00, -4.8640239E-02, 4.1764768E-05, -1.4452093E-08, 6.5459673E+00]
    Pv_w = np.exp((a[0]/TK) + a[1] + a[2]*TK + a[3]*TK**2 + a[4]*TK**3 + a[5]*np.log(TK))
    b = [-4.5818e-4, -2.0443e-6]
    return Pv_w * np.exp(b[0]*S + b[1]*S**2)

def SW_Viscosity2(T, S):
    # T in °C, S in PSU
    S1 = S / 1000.0
    a = [1.5700386464E-01, 6.4992620050E+01, -9.1296496657E+01, 4.2844324477E-05,
         1.5409136040E+00, 1.9981117208E-02, -9.5203865864E-05, 7.9739318223E+00,
        -7.5614568881E-02, 4.7237011074E-04]
    mu_w = a[3] + 1.0 / (a[0]*(T + a[1])**2 + a[2])
    A = a[4] + a[5]*T + a[6]*T**2
    B = a[7] + a[8]*T + a[9]*T**2
    return mu_w * (1.0 + A*S1 + B*S1**2)

def SW_Density2(T, S, P):
    # T in °C, S in PSU, P in MPa
    # Guard against T==0 in 1/T terms
    EPS_T = 1e-6
    Tsafe = np.where(np.abs(T) < EPS_T, EPS_T, T)

    P0 = SW_Psat2(T, S) / 1e6
    P0 = np.where(T < 100.0, 0.101325, P0)
    s = S / 1000.0

    a = [9.9992293295E+02, 2.0341179217E-02, -6.1624591598E-03, 2.2614664708E-05, -4.6570659168E-08]
    b = [8.0200240891E+02, -2.0005183488E+00, 1.6771024982E-02, -3.0600536746E-05, -1.6132224742E-05]
    rho_w = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

    invT  = 1.0 / Tsafe
    invT2 = invT * invT
    invT3 = invT2 * invT
    D_rho = b[0]*s + b[1]*s*invT + b[2]*s*invT2 + b[3]*s*invT3 + b[4]*s*s*invT2
    rho_sw = rho_w + D_rho

    c = [5.0792E-04, -3.4168E-06, 5.6931E-08, -3.7263E-10, 1.4465E-12, -1.7058E-15,
         -1.3389E-06, 4.8603E-09, -6.8039E-13]
    d = [-1.1077e-06, 5.5584e-09, -4.2539e-11, 8.3702e-09]
    F_P = np.exp(
        (P - P0) * (c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]*T**4 + c[5]*T**5 + S*(d[0] + d[1]*T + d[2]*T**2))
        + 0.5 * (P**2 - P0**2) * (c[6] + c[7]*T + c[8]*T**3 + d[3]*S)
    )
    return rho_sw * F_P

def SW_Kviscosity2(T, S):
    # T in °C
    P0 = SW_Psat2(T, S) / 1e6
    P0 = np.where(T < 100.0, 0.101325, P0)
    mu  = SW_Viscosity2(T, S)
    rho = SW_Density2(T, S, P0)
    return mu / rho

# ------------------ Wave/flux kernels (NumPy/xarray) ------------------

def _rename_latlon(ds):
    m = {}
    if "latitude"  in ds.coords: m["latitude"]  = "lat"
    if "longitude" in ds.coords: m["longitude"] = "lon"
    return ds.rename(m)

def wcm_u10(a, b, c, wind):
    # base >= 0 for fractional exponent
    base = xr.where(wind > c, wind - c, 0.0).astype("float64")
    return (a * (base ** b)).astype("float32")

def wcm_ust_hs_modelB(a, b, cp, ust, hs):
    den = xr.where(hs > 0, np.sqrt(9.8 * hs), np.nan).astype("float64")
    ratio = (ust.astype("float64") / den)
    ratio = xr.where(ratio >= 0, ratio, 0.0)
    w = a * cp.astype("float64") * (ratio ** b)
    w = xr.where(np.isfinite(w), w, np.nan)
    return w.astype("float32")

def kb_from_Va(Va, znotm, alpha, Sc, nu,
               r_min=1e-5, r_break=1e-3, r_max=1e-2, dr=2e-5, beta=1.5):
    eps = 1e-12
    Va_da = Va if isinstance(Va, xr.DataArray) else xr.DataArray(Va)

    r = xr.DataArray(np.arange(r_min, r_max + dr, dr), dims=("r",), name="r").astype("float64")
    q_small = (r**(-beta)) * (r_break**(beta - 10/3))
    q_large = r**(-10/3)
    qshape  = xr.where(r <= r_break, q_small, q_large)

    vol_kernel = (4.0/3.0) * np.pi * r**3
    norm = (vol_kernel * qshape).integrate("r")
    qshape = qshape / (norm + eps)

    alpha_s = xr.where(alpha > 0, alpha, np.nan).astype("float64")
    Sc_s    = xr.where(Sc    > 0, Sc,    np.nan).astype("float64")
    nu_s    = xr.where(nu    > 0, nu,    np.nan).astype("float64")

    D  = nu_s / (Sc_s + eps)
    chi = 9.81 * r**3 / (nu_s**2 + eps)
    yy  = 10.82 / (chi + eps)
    Ur0 = (2.0 * 9.81 * r**2) / (9.0 * (nu_s + eps)) * (np.sqrt(yy**2 + 2.0*yy) - yy)
    Ur  = xr.where(Ur0 < 0, 0, xr.where(Ur0 > 0.3, 0.3, Ur0))

    sqrt_arg = np.pi * D * Ur / (2.0 * r + eps)
    sqrt_arg = xr.where(sqrt_arg > 0, sqrt_arg, 0.0)
    kr = 8.0 * np.sqrt(sqrt_arg)

    Heqr = 4.0 * np.pi * r * Ur / (3.0 * alpha_s * (kr + eps))
    Er   = znotm / (Heqr + znotm + eps)
    Qr   = Va_da * qshape
    Vexch = (vol_kernel * Qr * Er).integrate("r")

    return (Vexch / (alpha_s + eps)).astype("float32")

# ------------------ Main (groupby-month) ------------------

import xarray as xr  # ensure after helpers (used above)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int, required=True, help="1..12")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    month = int(args.month)
    os.makedirs(args.outdir, exist_ok=True)

    # Paths
    dir_ml    = "/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/Analysis/NNWave/preds/"
    dir_truth = "/gpfs/f5/gfdl_o/scratch/Xiaohui.Zhou/Analysis/NNWave/dervied_cp_ust_2019_2022/"

    # Open WW3 truth/ML (NumPy-backed)
    ds_truth = xr.open_mfdataset(
        os.path.join(dir_truth, "ww3*.nc"),
        combine="by_coords", preprocess=_rename_latlon,
        chunks=None, parallel=False
    ).sortby("time")
    wcm_truth = ds_truth["wcm"]; u10 = ds_truth["wind"]; ust = ds_truth["ust"]; hs = ds_truth["hs"]; cp = ds_truth["cp"]

    ds_ml = xr.open_mfdataset(
        os.path.join(dir_ml, "ww3*.nc"),
        combine="by_coords", preprocess=_rename_latlon,
        chunks=None, parallel=False
    ).sortby("time")
    wcm_ml = ds_ml["wcm_pred"]

    # Monthly climatologies via groupby("time.month").mean("time")
    va_truth_m    = wcm_truth.groupby("time.month").mean("time").sel(month=month).load().astype("float32")
    va_ml_m       = wcm_ml   .groupby("time.month").mean("time").sel(month=month).load().astype("float32")
    va_u10_val_m  = wcm_u10(5.5e-06, 1.05, 2.35, u10).groupby("time.month").mean("time").sel(month=month).load().astype("float32")
    va_semi_val_m = wcm_ust_hs_modelB(2.4e-3, 2.48, cp, ust, hs).groupby("time.month").mean("time").sel(month=month).load().astype("float32")

    # SST (already in °C): monthly mean via groupby, then interpolate to Va grid
    sst_files = [
        "/gpfs/f5/gfdl_o/world-shared/datasets/reanalysis/JRA55-do/v1.5.0/padded/tos_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-5-0_gn_20190101-20191231.padded.nc",
        "/gpfs/f5/gfdl_o/world-shared/datasets/reanalysis/JRA55-do/v1.5.0/padded/tos_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-5-0_gn_20200101-20201231.padded.nc",
        "/gpfs/f5/gfdl_o/world-shared/datasets/reanalysis/JRA55-do/v1.5.0/padded/tos_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-5-0_gn_20210101-20211231.padded.nc",
        "/gpfs/f5/gfdl_o/world-shared/datasets/reanalysis/JRA55-do/v1.5.0/padded/tos_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-5-0_gn_20220101-20221231.padded.nc",
    ]
    ds_sst = xr.open_mfdataset(
        sst_files, combine="nested", concat_dim="time",
        data_vars="minimal", coords="minimal", compat="override", join="override",
        chunks=None, parallel=False
    ).sortby("time")

    tos_month  = ds_sst["tos"].groupby("time.month").mean("time").sel(month=month)  # °C
    tos_on_va  = tos_month.interp(lat=va_truth_m.lat, lon=va_truth_m.lon, method="linear").astype("float32").load()

    # Optional sanity
    try:
        units = ds_sst["tos"].attrs.get("units", "<none>")
    except Exception:
        units = "<none>"
    print(f"[month {month:02d}] SST units: {units}; mean={float(tos_on_va.mean()):.2f} °C", flush=True)

    # Thermophysical properties (S=35 PSU), array-mode apply_ufunc
    S_da = xr.full_like(tos_on_va, 35.0)

    alpha = xr.apply_ufunc(
        solco2_alpha, tos_on_va,
        input_core_dims=[[]], output_core_dims=[[]],
        dask="forbidden", output_dtypes=[np.float32]
    )
    Sc = xr.apply_ufunc(
        solco2_Sc, tos_on_va,
        input_core_dims=[[]], output_core_dims=[[]],
        dask="forbidden", output_dtypes=[np.float32]
    )
    nu = xr.apply_ufunc(
        SW_Kviscosity2, tos_on_va, S_da,
        input_core_dims=[[], []], output_core_dims=[[]],
        dask="forbidden", output_dtypes=[np.float32]
    )

    # z0m proxy from hs monthly mean
    hs_m  = hs.groupby("time.month").mean("time").sel(month=month).load().astype("float32")
    znotm = (hs_m / 2.0).astype("float32")

    # kb for each Va candidate
    kb_truth     = kb_from_Va(va_truth_m,    znotm, alpha, Sc, nu).rename("kb")
    kb_ml        = kb_from_Va(va_ml_m,       znotm, alpha, Sc, nu).rename("kb")
    kb_semi      = kb_from_Va(va_semi_val_m, znotm, alpha, Sc, nu).rename("kb")
    kb_u10_only  = kb_from_Va(va_u10_val_m,  znotm, alpha, Sc, nu).rename("kb")

    # Save per-product files for this month
    enc = {"kb": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    mm = f"{month:02d}"
    (kb_truth   ).to_dataset().to_netcdf(os.path.join(args.outdir, f"kb_truth_month{mm}.nc"),    encoding=enc)
    (kb_ml      ).to_dataset().to_netcdf(os.path.join(args.outdir, f"kb_ml_month{mm}.nc"),       encoding=enc)
    (kb_semi    ).to_dataset().to_netcdf(os.path.join(args.outdir, f"kb_semi_month{mm}.nc"),     encoding=enc)
    (kb_u10_only).to_dataset().to_netcdf(os.path.join(args.outdir, f"kb_u10_only_month{mm}.nc"), encoding=enc)

    print(f"[month {month:02d}] wrote kb_*_month{mm}.nc")

