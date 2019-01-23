"""
                Small script that changes names on misnamed files
"""
import os
import numpy as np

cwd = os.getcwd()
dir = cwd + "/LFP_files2/"

filenames = list(os.listdir(dir))

N = len(filenames)
for i in range(N):
    flnm = filenames[i].split("-")
    flnm[-1] = int(float(flnm[-1][:-4])*1000)

    new_filename = f"{flnm[0]}-{flnm[1]}-{flnm[2]}.npy"

    os.rename(dir+filenames[i], dir+new_filename)
