#%%
import numpy as np 
import matplotlib.pyplot as plt

from dan_way import distance_transform_np
x, y, z= np.arange(5), np.arange(5),np.arange(5)[:,np.newaxis, np.newaxis]
distance = np.sqrt(x ** 2 + y ** 2)
plt.pcolor(distance)
plt.colorbar()
plt.show()
# %%
easy = np.array([[0, 0, 1, 1, 0],
 [0, 1, 1, 0, 1],
 [1, 1, 1, 1, 0],
 [0, 1, 1, 1, 0],
 [0, 0, 1, 0, 0]])

#x, y = np.arange(5), np.arange(5)[:, np.newaxis]
#x, y = np.ogrid[2:7, 0:5]
x, y = np.mgrid[0:5, 0:5]
print(x)
print(y)
distance = np.sqrt((x-2)** 2 + (y-2) ** 2)
print(distance)

plt.pcolor(distance)
plt.colorbar()
plt.show()
# %%
    #def dans(mat):
    mat = easy
    m = mat.shape[0]
    n = mat.shape[1]
    x, y = np.mgrid[0:m, 0:n]
    ones = []
    for M in m:
        for N in n:
            if mat[M][N] == 1:
                ones.append(np.array([m,n]))
                x[0]

    print(mat,x,y)
    #print(no)
    distance = np.sqrt((x-3)** 2 + (y-2) ** 2)
    #print(distance)
    plt.pcolor(distance)
    plt.colorbar()
    plt.show()
    

# %%
