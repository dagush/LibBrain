# --------------------------------------------------------------------------------------
# WorkBrain base folders!
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
from sys import platform

if platform == "win32":
    WorkBrainFolder = "L:/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WorkBrain/"
elif platform == "darwin":
    WorkBrainFolder = "/Users/dagush/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WorkBrain/"
else:
    raise Exception('Unrecognized OS!!!')

WorkBrainDataFolder = WorkBrainFolder + "_Data_Raw/"
WorkBrainProducedDataFolder = WorkBrainFolder + "_Data_Produced/"