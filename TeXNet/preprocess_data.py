from typing_extensions import Self
import numpy as np
import scipy.io as scio
import os

# S_mu = []
# S_std = []

root = 'C:/Users/henrigur/OneDrive - NTNU/Documents/ProsjektOppg/HADAR database'
# Or if running from TeX-Net folder:
root = '../HADAR database/'
# path = '/home/sureshbs/Desktop/TeXNet/Dataset/HADAR_database/Scene14/'

# S_data = scio.loadmat(path+'HeatCubes/HSI.mat')["HSI"]
# print(np.shape(S_data))
# S_data = np.transpose(S_data, (2, 0, 1))
# # S_data = S_data[4:53] # last 39 channels)
# for i in  range(49):
#     mu = np.mean(S_data[i], (0, 1))
#     std = np.std(S_data[i], (0, 1), dtype=np.float64)
#     S_mu.append(mu)
#     S_std.append(mu)
# np.save(path+'HeatCubes/S_mu.npy', np.asarray(S_mu))
# np.save(path+'HeatCubes/S_std.npy', np.asarray(S_std))

# T_data = scio.loadmat(path+'GroundTruth/tMap/tMap.mat')["tMap"]
# T_mu = np.mean((T_data), (0, 1))
# T_std = np.std((T_data), (0, 1))
# np.save(path+'GroundTruth/tMap/T_mu.npy', np.asarray(T_mu))
# np.save(path+'GroundTruth/tMap/T_std.npy', np.asarray(T_std))

################ eList Processing #######################
ids = [f"L_{i:04d}" for i in range(1, 6)]
ids += [f"R_{i:04d}" for i in range(1, 6)]
SUBFOLDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SUBFOLDERS = ["Scene"+str(_) for _ in SUBFOLDERS]

print("eMap preprocessing")
for subfolder in SUBFOLDERS:
    print("Processing subfolder", subfolder)
    e_files = []
    elist_file = os.path.join(root, subfolder, 'GroundTruth',
                                            'eMap', f"eList.mat")
    for id in ids:
        e_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                            'eMap', f"eMap_{id}.mat"))
    for i in range(10):
        e_list = np.squeeze(np.asarray(scio.loadmat(elist_file)["eList"]))
        e_data = scio.loadmat(e_files[i])["eMap"]

        data = ((e_list[e_data - 1] - 1))
        if i<5:
            np.save(os.path.join(root, subfolder, "GroundTruth", "eMap", "new_eMap_L_000"+str(i+1)+".npy"),
                    np.asarray(data))
        else:
            np.save(os.path.join(root, subfolder, "GroundTruth", "eMap", "new_eMap_R_000"+str(i-4)+".npy"),
                    np.asarray(data))

### Commented out, Scene11 has different naming convention (in database) -H ###
""" ids = [f"L_{i:04d}" for i in range(1, 5)]
ids += [f"R_{i:04d}" for i in range(1, 5)]
SUBFOLDERS = [11]
SUBFOLDERS = ["Scene"+str(_) for _ in SUBFOLDERS]
for subfolder in SUBFOLDERS:
    print("Processing subfolder", subfolder)
    e_files = []
    elist_file = os.path.join(root, subfolder, 'GroundTruth',
                                            'eMap', f"eList.mat")
    for id in ids:
        e_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                            'eMap', f"eMap_{id}.mat"))
    for i in range(8):
        e_list = np.squeeze(np.asarray(scio.loadmat(elist_file)["eList"]))
        e_data = scio.loadmat(e_files[i])["eMap"]

        data = ((e_list[e_data - 1]-1))
        if i<4:
            np.save(os.path.join(root, subfolder, "GroundTruth", "eMap", "new_eMap_L_000"+str(i+1)+".npy"),
                    np.asarray(data))
        else:
            np.save(os.path.join(root, subfolder, "GroundTruth", "eMap", "new_eMap_R_000"+str(i-3)+".npy"),
                    np.asarray(data)) """
            
### Scene11 has different naming convention -H ###
print("eMap preprocessing for Scene11")
SUBFOLDERS = ["Scene11"]

for subfolder in SUBFOLDERS:
    print("Processing subfolder", subfolder)
    e_files = []
    elist_file = os.path.join(root, subfolder, 'GroundTruth', 'eMap', f"eList.mat")
    
    # Scene11 has files named eMap_0001.mat, eMap_0002.mat, etc.
    for i in range(1, 5):  # Only 4 frames in Scene11
        e_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                    'eMap', f"eMap_{i:04d}.mat"))
    
    e_list = np.squeeze(np.asarray(scio.loadmat(elist_file)["eList"]))
    
    for i in range(4):
        e_data = scio.loadmat(e_files[i])["eMap"]
        data = ((e_list[e_data - 1] - 1))
        
        # Save as both L and R (Scene11 doesn't distinguish, but dataloader expects both)
        np.save(os.path.join(root, subfolder, "GroundTruth", "eMap", 
                f"new_eMap_L_{i+1:04d}.npy"), np.asarray(data))
        np.save(os.path.join(root, subfolder, "GroundTruth", "eMap", 
                f"new_eMap_R_{i+1:04d}.npy"), np.asarray(data))
### End of Scene11 preprocessing -H ###

################### S_beta Processing ###################################################
import torch.nn.functional as F
import torch

# Separate for Scene 11
S_files = [os.path.join(root, "Scene11", "Radiance_EnvObj", f"S_EnvObj_{i:04d}.mat") for i in range(1, 5)]

for j in range(4):
    data = scio.loadmat(S_files[j])["S_EnvObj"]
    data = np.transpose(data)[np.newaxis, ..., np.newaxis]
    np.save(os.path.join(root, "Scene11", "HeatCubes", f"S_EnvObj_L_{j+1:04d}.npy"), data)
    np.save(os.path.join(root, "Scene11", "HeatCubes", f"S_EnvObj_R_{j+1:04d}.npy"), data)

SUBFOLDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SUBFOLDERS = ["Scene"+str(_) for _ in SUBFOLDERS]
ids = [f"L_{i:04d}" for i in range(1, 6)]
ids += [f"R_{i:04d}" for i in range(1, 6)]

print("S_beta preprocessing")
for subfolder in SUBFOLDERS:
    S_files = []
    print("processing subfolder", subfolder)
    for id in ids:
        S_files.append(os.path.join(root, subfolder, 'HeatCubes',
                                                 f"{id}_heatcube.mat"))

    for j in range(10):
        img = torch.tensor(np.asarray(scio.loadmat(S_files[j])["S"]))
        img = torch.permute(img, (2, 0, 1))
        img = torch.reshape(img, (1, 54, 1080, 1920))
        [b, c, h, w] = img.shape
        quadratic_split = F.avg_pool2d(img, (h//2, w))
        mean = quadratic_split.numpy()

        if j < 5:
            np.save(os.path.join(root, subfolder, "HeatCubes", "S_EnvObj_L_000"+str(j+1)+".npy"),
                    np.asarray(mean))
        else:
            np.save(os.path.join(root, subfolder, "HeatCubes", "S_EnvObj_R_000"+str(j-4)+".npy"),
                    np.asarray(mean))
