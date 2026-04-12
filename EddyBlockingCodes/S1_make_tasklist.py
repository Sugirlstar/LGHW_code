from itertools import product

regions = ["ATL", "NP", "SP"]
seasons = ["ALL", "DJF", "JJA"]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["AC", "CC"]
datasets = ["MERRA2", "JRA55", "ERA5"]

with open("/home/hu1029/Nature_Serendipity/EddyBlockingCodes/S1_tasklist.txt", "w") as f:
    for rgname, cyc, typeid, ss, dtname in product(regions, cycTypes, [1,2,3], seasons, datasets):
        f.write(f"{rgname} {cyc} {typeid} {ss} {dtname}\n")
