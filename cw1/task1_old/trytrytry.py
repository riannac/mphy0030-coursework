#%%
import numpy as np
import urllib.request
from scipy import ndimage
from matplotlib import pyplot as plt


#%%
#download
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

#%% 
#function def
################################################################################
#                            Function Definitions                              #
################################################################################
def print_properties(x):
    print("\n type : " + str(type(x))
    + "\n size : " + str(x.size)
    + "\n shape : " + str(x.shape)
    + "\n dtype : " + str(x.dtype)
    + "\n ndim : " + str(x.ndim)
    + "\n itemsize : " + str(x.itemsize)
    + "\n nbytes : " + str(x.nbytes))
    

def distance_transform_np(vol_bi_img):
    """docstring to describe the algorithm used"""
    #input: 3D volumetric bianary image
    #output: 3d euclidean distance transform
    print("this will be the transform")
    print(vol_bi_img)

def pre_built(vol_bi_img):
    """this is the in built distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)
def brute(vol_bi_img):
    return ndimage.distance_transform_bf(vol_bi_img)

"""visualisatino while developing comparing image and it's transform and specified slices"""
def plot_comp(data, slice_index=15):
    """plot image and its transform to compare"""
    #not perfect 
    plt.figure()
    plt.subplot(1,2,1)
    if data.ndim ==3:
        plt.imshow(data[slice_index,:,:], cmap=plt.cm.gray)
        #commutes full transform but only plots 1 slice
        plt.imshow(pre_built(data)[slice_index,:,:])
    else:
        plt.imshow(data, cmap=plt.cm.gray)
        plt.imshow(pre_built(data))
   
    plt.subplot(1,2,2)
    plt.show()


def alg_comp(data, slice_index=15):
    """plot my transform and the transform to compare"""
    plt.figure()
    plt.subplot(1,2,1)
    if data.ndim ==3:
        data = data[slice_index, : ,:] 
    plt.imshow(nearestOne(data))
    plt.subplot(1,2,2)
    plt.imshow(pre_built(data))
    plt.show()
#%%
#setting arrays
mdup_data = np.array([
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,1,1,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,1,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 ])
easy1 = np.array([
    [0,0,1,1,0],
    [0,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,1],
    [0,0,1,0,0]])

easy = np.array([[1,0,0,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1]])

#def nearest(mat):
#%%
    mat = mdup_data
    MAX = 1000
    INT_MAX = 9999
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    
    newx = [-1, 0, 1, 0]#, -1,1,1,1]
    newy = [0, -1, 0, 1]#,-1,1,-1,1]
   
    q = []
     
    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            euc[i][j] = INT_MAX
            
            if (mat[i][j] == 0):
                dist[i][j] = 0
                euc[i][j] = 0
                q.append([i, j])
    
    poped = []
    while (len(q)):
        poped = q[0]
        q.pop(0)
            
        x = poped[0]
        y = poped[1]
          
    

        for i in range(len(newx)):
            adjx = x + newx[i]
            adjy = y + newy[i]
                  
            
            
            if dist[x][y] == 0:
                this = x
                that = y

            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):
                e = np.sqrt((this-adjx)**2+(that-adjy)**2)
                
                if euc[adjx][adjy] > e:
                    euc[adjx][adjy] = e
                q.append([adjx, adjy])
                #eq = np.sqrt((this-(x+1))**2+((that-y)**2))
                #ew = np.sqrt((this-x)**2+(that-(y+1))**2)
                #print (e,eq,ew)
                if  dist[adjx][adjy] > dist[x][y] + 1:
                    dist[adjx][adjy] = dist[x][y] + 1

                  #  if e > eq:
                   #     euc[adjx][adjy] = eq
                        #q.append([adjx, adjy])

                    #elif e > e:
                     #   euc[adjx][adjy] = ew
                        #q.append([adjx, adjy])

                    #eq = np.sqrt((this-adjy)**2+(that-adjx)**2)
                    #print("adjx,adjy: " + str((adjx,adjy)))
                    #print("x,y: " + str((x,y)), end = '')
                    
                    #euc[adjx][adjy] = e 
                

#                if euc[adjx][adjy] > e:

                #if euc[adjx][adjy] > e :
                    #print("w")
                 #   euc[adjx][adjy] = e
                #if euc[adjx][adjy] > eq:
                    #print("h")
                    #euc[adjx][adjy] = eq
                    #
                #elif e < eq:
                    #euc[adjx][adjy] = eq
                #if e > euc[adjy][adjy]
 #               if  dist[adjx][adjy] > dist[x][y] + 1:
                    ##print(dist[adjx][adjy],end =" ")
                    #print(dist[x][y],end =" ")
                    #e = np.sqrt((x-adjx)**2+(y-adjy)**2)
                    #print("adjx,adjy: " + str((adjx,adjy)),end =" ")
                                       
                    #dist[adjx][adjy] > print(x,y)
                    #print(dis_i_need(x,adjx,y,adjy))
                    #euc[adjx][adjy] = dis_i_need(adjx,x,adjy,y)
                    #euc[adjx][adjy] = dis_i_need(x,adjx,y,adjy)

                  #  dist[adjx][adjy] = dist[x][y] + 1
                   # q.append([adjx, adjy])
                    #euc[adjx][adjy] = e
                    
                    #print(adjx,adjy,x,y)
                    #np.sqrt(((x-adjx)**2)+((y-adjy)**2))
    #print((dist))
    print("\n")
    #print(euc)
    #print(pre_built(mat))
    #return(euc)
    #return np.sqrt(dist)
    #print(euc)

#mat = mdup_data
#print(pre_built(mat))
#print(nearest(mat))
        
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(pre_built(mat))
    plt.subplot(1,3,2)
    plt.imshow(euc)
    plt.subplot(1,3,3)
    plt.imshow(dist)
    plt.show()


# %%

                #if dist[adjx][adjy] == 0:
                 #   this = adjx
                  #  that = adjx
                #print((this,that, end ='')
                #e = np.sqrt((this-adjx)**2+(that-adjy)**2)
                #print(e, end=' ')