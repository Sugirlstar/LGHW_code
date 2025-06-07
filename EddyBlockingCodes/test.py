
import pickle


with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLonList_blkType1_ATL_ALL", "rb") as fp:
    centerLon1 = pickle.load(fp)

print(centerLon1)

print('done')