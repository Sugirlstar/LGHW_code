import os
import math
import numpy as np
import xarray as xr
from math import pi
import multiprocessing as mp

from LWA_f2 import eqlat, lwa

# Global variables for workers
# ---------------------------
Z_ALL = None
LAT = None
LON = None
AREA = None
DPHI = None
EQ1 = None
EQ2 = None
NLAT = None
NLON = None
COSLAT = None


def init_worker(z_all, lat, lon, area, dphi, eq1, eq2, coslat):
    """
    Initialize global variables for each worker.
    """
    global Z_ALL, LAT, LON, AREA, DPHI, EQ1, EQ2, NLAT, NLON, COSLAT

    Z_ALL = z_all
    LAT = lat
    LON = lon
    AREA = area
    DPHI = dphi
    EQ1 = eq1
    EQ2 = eq2
    NLAT = len(lat)
    NLON = len(lon)
    COSLAT = coslat


def calc_one_t(t):
    """
    Calculate LWA for one time index t.
    Returns:
        t (int), LWA_z, LWA_z_A, LWA_z_C (2d arrays of shape (nlat, nlon))
    """
    global Z_ALL, LAT, LON, AREA, DPHI, EQ1, EQ2, NLAT, NLON, COSLAT

    z2d = Z_ALL[t, :, :]   # shape = (nlat, nlon), float32

    # allocate outputs for this time slice
    LWA_z = np.zeros((NLAT, NLON), dtype=np.float32)
    LWA_z_A = np.zeros((NLAT, NLON), dtype=np.float32)
    LWA_z_C = np.zeros((NLAT, NLON), dtype=np.float32)

    # ---------------- NH ----------------
    lat1 = LAT[EQ1::-1]
    nlat1 = len(lat1)
    dphi1 = DPHI[EQ1::-1, :]   # shape (nlat1, nlon)
    _, laa1 = np.meshgrid(LON, lat1)

    q_part1 = eqlat(
        z2d[EQ1::-1, :],
        AREA[EQ1::-1, :],
        LAT[EQ1::-1],
        1
    )
    LWA_z1, LWA_z1_A, LWA_z1_C = lwa(
        z2d[EQ1::-1, :],
        q_part1,
        nlat1,
        NLON,
        laa1,
        lat1,
        dphi1,
        1
    )

    # ---------------- SH ----------------
    lat2 = LAT[-1:-EQ2:-1]
    nlat2 = len(lat2)
    dphi2 = DPHI[-1:-EQ2:-1, :]
    _, laa2 = np.meshgrid(LON, lat2)

    q_part2 = eqlat(
        z2d[-1:-EQ2:-1, :],
        AREA[-1:-EQ2:-1, :],
        LAT[-1:-EQ2:-1],
        2
    )
    LWA_z2, LWA_z2_A, LWA_z2_C = lwa(
        z2d[-1:-EQ2:-1, :],
        q_part2,
        nlat2,
        NLON,
        laa2,
        lat2,
        dphi2,
        2
    )

    # merge NH + SH
    LWA_z[0:EQ1+1, :] = LWA_z1[::-1, :]
    LWA_z[EQ2:NLAT, :] = LWA_z2[::-1, :]

    LWA_z_A[0:EQ1+1, :] = LWA_z1_A[::-1, :]
    LWA_z_A[EQ2:NLAT, :] = LWA_z2_A[::-1, :]

    LWA_z_C[0:EQ1+1, :] = LWA_z1_C[::-1, :]
    LWA_z_C[EQ2:NLAT, :] = LWA_z2_C[::-1, :]

    # divide by cos(lat) for total LWA only
    LWA_z = LWA_z / COSLAT[:, None]

    print(f"Time index {t} finished", flush=True)

    return t, LWA_z, LWA_z_A, LWA_z_C


def prepare_static_fields(lat, lon):
    """
    Compute static fields that only depend on lat/lon.
    Returns:
        area, dphi, eq1, eq2, coslat
    """
    nlat = len(lat)
    nlon = len(lon)

    eq1 = int(nlat / 2) - 1
    eq2 = nlat - eq1 - 1

    dlat = (lat[0] - lat[1]) * pi / 180.0
    dlon = (lon[1] - lon[0]) * pi / 180.0

    R = 6.378e6

    clat0 = np.cos(lat * pi / 180.0).astype(np.float32)
    clat = np.abs(clat0[:, None] * np.ones((nlat, nlon), dtype=np.float32))
    dphi = (R * dlat * clat).astype(np.float32)

    area = np.zeros((nlat, nlon), dtype=np.float32)
    for la in range(nlat):
        if lat[la] == 90 or lat[la] == -90:
            area[la, :] = (R**2) * (1 - np.sin(pi / 2 - dlat / 2)) * dlon
        else:
            area[la, :] = (R**2) * (
                np.sin(lat[la] * pi / 180.0 + dlat / 2) -
                np.sin(lat[la] * pi / 180.0 - dlat / 2)
            ) * dlon

    coslat = np.cos(np.deg2rad(lat)).astype(np.float32)

    return area, dphi, eq1, eq2, coslat


def load_dataset_to_numpy(filepath, lat_name, lon_name, time_name, var_name, dtname):
    """
    Read the entire dataset into memory as float32 numpy array.
    Output z_all shape = (ntime, nlat, nlon), lat decreasing (north->south)
    """
    ds = xr.open_dataset(filepath)

    da = ds[var_name].squeeze()

    if dtname == "ERA5":
        # ERA5 geopotential -> geopotential height (m)
        da = da / 9.80665

    # sort latitude descending: 90 -> -90
    da = da.sortby(lat_name, ascending=False)

    lat = da[lat_name].values
    lon = da[lon_name].values

    # force in-memory numpy array
    z_all = da.values.astype(np.float32)

    ds.close()

    if z_all.ndim != 3:
        raise ValueError(f"Expected 3D array (time, lat, lon), got shape {z_all.shape}")

    ntime, nlat, nlon = z_all.shape
    print(f"Loaded data: time={ntime}, lat={nlat}, lon={nlon}", flush=True)
    print(f"z_all dtype={z_all.dtype}, size={z_all.nbytes/1024**3:.2f} GB", flush=True)

    return z_all, lat, lon


def Cal_parallel_inmemory(filepath, out_dir, dtname, yearname,
                          lat_name='lat', lon_name='lon', time_name='time',
                          var_name='z', nproc=64, chunksize=1):
    """
    Main driver:
    1) read full dataset into memory
    2) precompute static fields
    3) parallelize over time index
    4) save outputs
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- load full data into memory ----
    z_all, lat, lon = load_dataset_to_numpy(
        filepath=filepath,
        lat_name=lat_name,
        lon_name=lon_name,
        time_name=time_name,
        var_name=var_name,
        dtname=dtname
    )

    ntime, nlat, nlon = z_all.shape

    # ---- static fields ----
    area, dphi, eq1, eq2, coslat = prepare_static_fields(lat, lon)

    # ---- output arrays ----
    LWA_td = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    LWA_td_A = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    LWA_td_C = np.zeros((ntime, nlat, nlon), dtype=np.float32)

    # ---- multiprocessing ----
    # On Bell/Linux, fork is usually the fastest for this use case
    ctx = mp.get_context("fork")

    with ctx.Pool(
        processes=nproc,
        initializer=init_worker,
        initargs=(z_all, lat, lon, area, dphi, eq1, eq2, coslat)
    ) as pool:
        for t, lwa_, lwa_a_, lwa_c_ in pool.imap_unordered(calc_one_t, range(ntime), chunksize=chunksize):
            LWA_td[t, :, :] = lwa_
            LWA_td_A[t, :, :] = lwa_a_
            LWA_td_C[t, :, :] = lwa_c_

    # ---- convert to lat increasing order before save ----
    LWA_td = LWA_td[:, ::-1, :]
    LWA_td_A = LWA_td_A[:, ::-1, :]
    LWA_td_C = LWA_td_C[:, ::-1, :]
    lat_out = lat[::-1]

    # # ---- save ----
    np.save(f"{out_dir}/{dtname}_LWA_td_{yearname}_6hr.npy", LWA_td)
    np.save(f"{out_dir}/{dtname}_LWA_td_A_{yearname}_6hr.npy", LWA_td_A)
    np.save(f"{out_dir}/{dtname}_LWA_td_C_{yearname}_6hr.npy", LWA_td_C)
    np.save(f"{out_dir}/{dtname}_LWA_lat_{yearname}_6hr.npy", lat_out)
    np.save(f"{out_dir}/{dtname}_LWA_lon_{yearname}_6hr.npy", lon)

    # print("All done and files saved.", flush=True)

    #%% Plot
    import matplotlib.pyplot as plt
    Plot = np.nanmean(LWA_td, axis=0)  # Sum over time to get total LWA
    fig = plt.figure(figsize=[12,7])
    plt.contourf(lon,lat_out,Plot, 15, extend="both", cmap='Reds') 
    cb=plt.colorbar()

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    plt.savefig(f"{dtname}_LWAtotalMean_{yearname}_Parallel.png")
    plt.close()

    # delete the arrays to save memory
    del LWA_td, LWA_td_A, LWA_td_C, lat, lon



def main():

    OUT_DIR_List = {
        "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5",
        "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2",
        "JRA3Q": "/scratch/bell/hu1029/LGHW/interm_JRA3Q"
    }
    timerefFile = {
        "ERA5": "/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2025_1dg.nc",
        "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_Z500_6hr_1980_2025_1dg.nc",
        "JRA3Q": "/scratch/bell/hu1029/Data/processed/JRA3Q_Z500_6hr_1979_2025_1dg.nc"
    }
    varnameList = {
        "ERA5": "z",
        "MERRA2": "H",
        "JRA3Q": "hgt-pres-an-ll125"
    }
    yearnameList = {
        "ERA5": "1979_2025",
        "MERRA2": "1980_2025",
        "JRA3Q": "1979_2025"
    }

    # choose dataset(s)
    dtname = "ERA5"
    out_dir = OUT_DIR_List[dtname]
    yearname = yearnameList[dtname]
    filepath = timerefFile[dtname]
    var_name = varnameList[dtname]

    lat_name, lon_name, time_name = "lat", "lon", "time"

    # number of worker processes
    nproc = 128
    chunksize = 1

    Cal_parallel_inmemory(
        filepath=filepath,
        out_dir=out_dir,
        dtname=dtname,
        yearname=yearname,
        lat_name=lat_name,
        lon_name=lon_name,
        time_name=time_name,
        var_name=var_name,
        nproc=nproc,
        chunksize=chunksize
    )

if __name__ == "__main__":
    main()

