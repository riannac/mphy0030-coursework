#%%
from classes_etc import trans
from classes_etc import images
from classes_etc import rand_im
from classes_etc import rand_t
from classes_etc import r_scal
from classes_etc import r_scal_t
from classes_etc import plot_slices
from classes_etc import pil_silce
from classes_etc import lbt_data
from classes_etc import dims
#%%
from scipy.interpolate import interpn
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import os

#%%
#lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/image_train00.npy'
#urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
#%%
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

# %%
# %%

lbt_data = np.load('lbt_file.npy',allow_pickle=False)
im = lbt_data
dims = np.array([2,.5,.5])
c =lbt_data.shape[0]
a =lbt_data.shape[1]
b =lbt_data.shape[2]

def func(image,z, x,y):
    return image[z,x,y]
x = np.arange(0, a)*dims[1]+.5
y = np.arange(0, b)*dims[2]+.5
#y = np.arange(0, c)
#grid_x, grid_y = np.mgrid[ 0:a, 0:b ]
grid_x, grid_y = np.meshgrid(x,y)

#%%
rng = np.random.default_rng()

xpts = rng.choice(a, size=4000) #, replace = False)
ypts = rng.choice(b, size=4000)#, replace = False)
points = (xpts*dims[1], ypts*dims[2])
#%%
from scipy.interpolate import griddata

def p_slice(im, t, title, z):
    values = func(im, z, xpts[:], ypts[:])
    gris_z0 = griddata(points, values, (grid_x,grid_y), method='linear')
    plt.imshow(gris_z0.T, origin='lower', aspect=(1))
    plt.xticks(np.arange(0,128,24),np.arange(0,128,24)*dims[1])
    plt.yticks(np.arange(0,128,24),np.arange(0,128,24)*dims[1])
    plt.title(title)
    plt.savefig("../task3/"+t +"_s" + str(z))
#p_slice(lbt_data,"slice","Translation", 15)
#%%
slist = [5,10,15,20,25]
for e in range(2):
    for s in slist:
        p_slice(images[e],"Man_trans_vis"+str(e),"Manual Translation",s)
        p_slice(rand_im[e],"Ran_trans_vis"+str(e),"Random Translation",s)
        p_slice(r_scal[e], "Ran_scal_vis"+str(e),"Random Translation with strength parameter",s)

