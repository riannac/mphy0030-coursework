#%%

from email.errors import BoundaryError
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
#print(images)
title = "man_trans"
#plot_slices(images[1:2], title='../task3/'+title+str(0), slist = [3,9,17,22,25])
slist = [3,9,17,22,25]
#plot_slices(images,"Manual_trans_/Manual Transformation",slist)
#plot_slices(rand_im,"Random_trans_/Random transforms",slist)
#plot_slices(r_scal,"Rand_sal_trans_/Strength Parameter",slist)

def plot_new(images, t, pltt, slist):
    dir = "../task3/"+ t
    os.mkdir(dir)
    num = 0
    for image in images:
        c=0
        t_i = t+str(c)
        fig, ax = plt.subplots(1, 5)
        fig.set_size_inches(10,2.5) 
        for slice in slist:
            array = image[slice]
            ax[c].imshow(array, cmap='gray',origin='lower', vmin=0, vmax=1)
            ax[c].title.set_text('Depth = ' + str(slice))
            c += 1
            title = dir+"/"+t_i + "_s"+str(slice)+".png"
        num += 1 
        plt.suptitle(pltt + str(num))
        plt.savefig('../task3/'+title+str(num)+'.png', transparent = False)
        plt.show()
plot_new(images [1:2],    "Manual_trans_vis", "Manual Transform " ,slist)
plot_new(rand_im[1:2],   "Random_trans_vis", "Random Transform ",slist)
plot_new(r_scal [1:2],    "Rand_sal_trans_vis", "Varying Strength Parameter in Random Transform ",slist)
#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
#%%  

#%%
"""
# %%
lbt = lbt_data[15]
a =lbt[:][0]
b =lbt[0][:]
print(a.shape)

#c =lbt_data.shape[2]
#%%

XX, YY = np.meshgrid(a, b, sparse=True)

#%%
coord = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1), ZZ.reshape(-1,1)), axis=1)
maskx = np.random.random(a) <0.3
masky = np.random.random(b) <0.3
maskz = np.random.random(c) <0.2

ipts = maskx* x
jpts = masky* y
kpts = maskz* z

points = (ipts, jpts, kpts)

image_interpn_flatten = interpn(points, lbt_data, coord, bounds_error=False, fill_value=0)
image_interpn = image_interpn_flatten.reshape(lbt_data.shape)
warp = image_interpn

#%%
x = #np.arange(0, a)
y = np.arange(0, b)
z = np.arange(0, c)
XX, YY= np.meshgrid(x, y, indexing='ij')#, sparse=True)

maskx = np.random.random(a) <0.3

#%%

fig = plt.figure()
#ax = fig.gca(projection='3d')
#Z = images[0]
#surf = ax.plot_surface(XX[0], YY[0], ZZ[1], rstride=1, cstride=1,
 #   linewidth=0, antialiased=False)

#%%
plt.plot(XX,YY)
print(XX)
#%%
# %%
"""