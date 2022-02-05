#############################################################
#           2d WORKING ! as of 09.01.22
#           3d WORKING ! but very very slow 
#               but not pretty
##########################################################
# DO NOT EDIT
# 09/01/22
# was new_new_diff

"""trying from the start with new method""" 

"""
Task.py should perform: 
download label train from;
https:weisslab.cd.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy

implement distance_transform_np
input: 3D volumetric bianary image
output: 3d euclidean distance transform

"""
#%%
#import statements 
import numpy as np
import time
#%%
#importing data

""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

#%%
#defining functin

def pre_built(vol_bi_img):
    """this is the in built distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)

#%%
#setting arrays
easy = np.array([
    [0,0,1,1,0],
    [0,1,1,0,1],
    [1,1,1,1,0],
    [0,1,1,1,0],
    [0,0,1,0,0]])

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

easy3d = np.array([
        [[ 0, 1, 1, 1, 0],
        [ 1, 1, 1, 1, 1],
        [ 0, 1, 1, 1, 0],
        [ 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]],

       [[ 0, 0, 1,1,0],
        [0, 0, 1,1,1],
        [1, 0, 1,1,0],
        [0,0,0,1,1],
        [0,0,0,0,1]],

       [[0, 0, 0,1,1],
        [0, 0, 1,1,1],
        [1, 0, 0,1,0],
        [0,0,0,1,1],
        [0,0,1,1,1]]])



#%%
def printDistance(mat):
    N = mat.shape[0]
    M = mat.shape[1]
    ans = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    
    # Initialize the answer matrix
    # with INT_MAX.
    for i in range(N):
        for j in range(M):
            ans[i][j] = 999999999999
            euc[i][j] = 999999999999
    
    # For each cell 
    for i in range(N):
        for j in range(M):
            
            # Traversing the whole matrix 
            # to find the minimum distance.
            for k in range(N):
                for l in range(M):
                    
                    # If cell contain 0, check 
                    # for minimum distance. 
                    if (mat[k][l] == 0):
                        print(ans[i][j])
                        ans[i][j] = min(ans[i][j], 
                                    abs(i - k) + abs(j - l))
                        euc[i][j] = min(euc[i][j], 
                                    np.sqrt((((i-k)**2))+((j-l)**2)))
       # Printing the answer.
   
    #return euc
#%%
def printDistance(mat):
    N = mat.shape[0]
    M = mat.shape[1]
    ans = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    
    # Initialize the answer matrix
    # with INT_MAX.
    for i in range(N):
        for j in range(M):
            ans[i][j] = 999999999999
            euc[i][j] = 999999999999
    
    # For each cell 
    for i in range(N):
        for j in range(M):
            
            # Traversing the whole matrix 
            # to find the minimum distance.
            for k in range(N):
                for l in range(M):
                    
                    # If cell contain 0, check 
                    # for minimum distance. 
                    if (mat[k][l] == 0):
                    
                        ans[i][j] = min(ans[i][j], 
                                    abs(i - k) + abs(j - l))
                        euc[i][j] = min(euc[i][j], 
                                    np.sqrt((((i-k)**2))+((j-l)**2)))
       # Printing the answer.
    return euc


def printDistance3d(mat):
    N = mat.shape[0]
    M = mat.shape[1]
    L = mat.shape[2]
    euc = np.zeros(shape = mat.shape)
    
    # Initialize the answer matrix
    # with INT_MAX.
    for i in range(N):
        for j in range(M):
            for k in range(L):    
                euc[i][j][k] = 999999999999
    
    # For each cell 
    for i in range(N):
        for j in range(M):
            for k in range(L):
            # Traversing the whole matrix 
            # to find the minimum distance.
                for n in range(N):
                    for m in range(M):
                        for l in range(L):
                            # If cell contain 0, check 
                            # for minimum distance. 
                            if (mat[n][m][l] == 0):
                            
                                euc[i][j][k] = min(euc[i][j][k], 
                                            np.sqrt((((i-n)**2))+((j-m)**2) +((k-l)**2)))
       # Printing the answer.
    return euc
    #return np.sqrt(ans)
#    for i in range(N):
 #       for j in range(M):
  #          print(ans[i][j], end = " ")
   #     print()

# Driver Code
N = 3
M = 4
mat = np.array([[0, 0, 0, 1], 
       [0, 0, 1, 1],
       [0, 1, 1, 0]])


#%%
my_alg_time = time.time()
a1 =printDistance3d(lbt_data)
print("--- %s seconds ---" % (time.time() - my_alg_time))

pre_alg_time = time.time()
a2 = pre_built(lbt_data)
print("--- %s seconds ---" % (time.time() - pre_alg_time))

#a1 = printDistance(easy)
#a2 = pre_built(easy)

#print(a1)
#print('\n')
#print(a2)

print(np.array_equal (a1, a2, equal_nan=False))
print(np.allclose(a1,a2, rtol=.2, atol=.2))

#print(mdup_data)
#nearestOne(mdup_data)
plt.subplot(1,2,1)
plt.imshow(a1[2])
plt.subplot(1,2,2)
plt.imshow(a2[2])
plt.show()



# This code is contributed by PranchalK
# %%
