import numpy as np
raw_data = np.load("/home/ajivani/WLROM_new/WhiteLight/validation_data/CR2161_validation_PolarTensor.npy")
# simIDs = np.load("/home/ajivani/WLROM_new/WhiteLight/validation_data/CR2161_SimID4edge_validation.npy")
simIDs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
raw_data.shape

sample_rd = raw_data[:126, :, 25:, 20]
nr, ntheta, nt = sample_rd.shape
print(nr, ntheta, nt)

# %%

r_lower = 3.903302085636277
r_upper = 23.465031329617336
theta = np.linspace(0, 2 * np.pi, 512)
r = np.linspace(r_lower, r_upper, 126)
t = np.linspace(0, 0.015625 * (nt - 1), nt)


# %%
import os
for sim_idx, sim in enumerate(simIDs):
    sample_rd = raw_data[:126, :, 25:, sim_idx]
    os.makedirs('Intensities_Sim_idl_format/CR2161_Sim{:03d}'.format(sim), exist_ok=True)
    for ttt in range(nt):
        print("Saving for timestep ", ttt, " sim ", sim)
        with open('Intensities_Sim_idl_format/CR2161_Sim{:03d}/intensities_t{:03d}.txt'.format(sim, ttt), 'w') as f:
            f.write(f"White Light Intensity Observations\n")
            f.write(f"{ttt + 1} {t[ttt]} 2 0 1\n")
            f.write(f"{nr} {ntheta}\n")
            f.write("R Theta Intensity\n")
            for j in range(ntheta):
                for i in range(nr):
                    f.write(f"{r[i]:.6f} {theta[j]:.6f} {sample_rd[i,j, ttt]:.6f}\n")

    print("ASCII file has been created successfully.")

# %%


