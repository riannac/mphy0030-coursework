
#%%
#importing
"""THis is WOKRINGGGGGG"""
import numpy as np
import numpy as np
import time
import urllib.request
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_bf
from matplotlib import pyplot as plt
#%%
#setting arrays
easy = np.array([
    [1,0,0,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,1],
    [0,1,1,1,1]])

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


mdup_data3d = np.array([
                [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],
                 
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 0, 0, 0,1,1,1,1,1,0],
                 [0, 1, 1, 1, 0, 0, 0, 0, 0,0,0,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 0, 0, 0,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
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
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],

                
                 ])

#%%
#prebuilt
def pre_built(vol_bi_img):
    """this is the scipy distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)


#%%
#best2d
def best(mat):
    """it was working but i've managed to fuck it up"""
    #mat = mdup_data
    m = mat.shape[0]
    n = mat.shape[1]
    newx = [-1, 0, 1, 0]#1, 1,-1,-1]
    newy = [0, -1, 0, 1]#1,-1,-1, 1]
     
    ones=[]
    zeros=[]
    for i in range(m):
        for j in range(n): 
            if mat[i][j]==1:
                ones.append(np.array([i,j]))
            else: 
                zeros.append(np.array([i,j]))
                #q.append([i,j])
                
    #print(easy)
    #ones = [np.array([2,2]), np.array([0,3])]
    euc_array= np.zeros(shape=(mat.shape))
    #print(ones)
    visit= np.ones(shape=(mat.shape))
    visit = visit*999
       
    for one in ones:
        
        visit[one[0]][one[1]] = 0
        #euc_array[one[0]][one[1]] = 0
        #euc_array = euc_array*999
        #print((one[0],one[1]))
        q= [one]
        euc_dist=[]
        poped = []

        while (len(q)):
            poped = q[0]
            q.pop(0)

            # coordinate of currently popped node
            x = poped[0]
            y = poped[1]
            
            #print(poped, end ='')
       
            # now check for all adjancent of popped element
            for i in range(4):
                
                adjx = x + newx[i]
                adjy = y + newy[i]
                #print([x,y], end='')
                if (adjx >= 0 and adjx < m and adjy >= 0 and
                    adjy < n):
                    
                    e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                    if mat[adjx][adjy]==0:
                        euc_dist.append(e)
                        
                    elif visit[adjx][adjy] > visit[x][y] +1:
                        visit[adjx][adjy] = visit[x][y] +1
                        # == 0:# visit[adjx][adjy] + 1:
                        #visit[adjx][adjy] = 1
                        #print((adjx,adjy),end='')
                    
                        #if mat[adjx][adjy]==0:
                         #   e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                          #  euc_dist.append(e)
                           # print("zero at"+str((adjx,adjy)))
                            #print(min(euc_dist)) 
                            #visit[adjy][adjy] = min(euc_dist)
                        #else:
                        q.append([adjx,adjy])
                     #   visit[adjy][adjy] = 1
                    #else:
                       # q.append([adjx, adjy])
        euc_array[one[0]][one[1]] =  min(euc_dist)
        #print(visit)
    #print(visit)

    return euc_array

#%%
#dansmat
def dans(mat):
    #mat = mdup_data
    m = mat.shape[0]
    n = mat.shape[1]

    ones=[]
    zeros=[]
    for i in range(m):
        for j in range(n): 
            if mat[i][j]==0:
                zeros.append(np.array([i,j]))
            else: 
                ones.append(np.array([i,j]))
    
    #print(easy)
    print(len(zeros))
    print(len(ones))
    """
    euc_array = np.zeros(shape=(mat.shape))
    #ones = [np.array([0,2]), np.array([0,3])]
    for one in ones:
        euc_dist=[]
        for zero in zeros:
            dist = np.sqrt((one[0]-zero[0])**2+(one[1]-zero[1])**2)
            euc_dist.append(dist)
        euc_array[one[0]][one[1]] = min(euc_dist)
        #print(min(euc_dist))
        #print(euc_dist.index(min(euc_dist)))
        
        #print(min(euc_dist))
    return euc_array"""
#%%
def best3d(mat):
#%%
    #mat = mdup_data3d
    mat = easy3d
    """editing"""
    #mat = mdup_data
    m = mat.shape[0]
    n = mat.shape[1]
    l = mat.shape[2]
    newx = [-1, 0, 0, 1,0,0]
    newy = [0, -1, 0, 0,1,0]
    newz = [0, 0, -1, 0, 0,1]
    ones=[]
    zeros=[]
    
    euc_array= np.zeros(shape=(mat.shape))
    visit= np.ones(shape=(mat.shape))
    visit = visit*9999
    print(mat)
    for i in range(m):
        for j in range(n): 
            for k in range(l):
                if mat[i][j][k]==1:
                    ones.append(np.array([i,j,k]))
                else: 
                    zeros.append(np.array([i,j,k]))
                    visit[i][j][k] = 0
                    
                
    ones = [np.array([0, 0, 1]), np.array([0, 0,2])]
    
    for one in ones:
        visit[one[0]][one[1]][one[2]] = 1
        print((one), end = ' ')
        
        q= [one]
        euc_dist=[]
        poped = []
        while (len(q)):
            poped = q[0]
            q.pop(0)

            # coordinate of currently popped node
            x = poped[0]
            y = poped[1]
            z = poped[2]

            # now check for all adjancent of popped element
            for i in range(len(newx)):
                
                adjx = x + newx[i]
                adjy = y + newy[i]
                adjz = z + newz[i]
                #print([x,y], end='')
                if (adjx >= 0 and adjx < m and
                     adjy >= 0 and adjy < n and 
                      adjz >= 0 and adjz < l):
                    #if visit[adjx][adjy][adjz] == 0:
                    #print([adjx, adjy,adjz], end = ' ')
                    
                    e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2+(one[2]-adjz)**2)
                    if mat[adjx][adjy][adjz]==0:
                            euc_dist.append(e)
                        
                    if mat[adjx][adjy][adjz]==1:
                        visit[adjx][adjy][adjz] = visit[x][y][z]+1
                        q.append([adjx,adjy,adjz])
                        #print("were")  #
                    #if visit[adjx][adjy][adjz] > visit[x][y][z] +1:
                     #   visit[adjx][adjy][adjz] = visit[x][y][z] +1
                        q.append([adjx,adjy,adjz])

                        if mat[adjx][adjy][adjz]==0:
                            euc_dist.append(e)
                        #print("here")
                        
            print (visit)
                        #print("here")
            print(euc_dist)
    #print(euc_dist)

    #print(min(euc_dist))

#    euc_array[one[0]][one[1]][one[2]] =1
#    print(euc_array[one[0]][one[1]][one[2]])


                     #   visit[adjy][adjy] = 1
                    #else:
                       # q.append([adjx, adjy])
            #euc_array[one[0]][one[1]][one[2]] =  min(euc_dist)
        #print(visit)

    #print(euc_array)
    #print(pre_built(mat))

#%%
#editing dan2d
#%%
    #def best(mat):
    
    mat = easy
    m = mat.shape[0]
    n = mat.shape[1]
    newx = [-1, 0, 1, 0]#1, 1,-1,-1]
    newy = [0, -1, 0, 1]#1,-1,-1, 1]
     
    ones=[]
    zeros=[]
    for i in range(m):
        for j in range(n): 
            if mat[i][j]==1:
                ones.append(np.array([i,j]))
            else: 
                zeros.append(np.array([i,j]))
                #q.append([i,j])
                   
    #ones = [np.array([2,2]), np.array([0,3])]
    euc_array= np.zeros(shape=(mat.shape))
    #print(ones)
    visit= np.zeros(shape=(mat.shape))
    visit = visit*999
    
    for one in ones:
        visit[one[0]][one[1]] = 1
    #print(visit)


        #euc_array[one[0]][one[1]] = 0
        #euc_array = euc_array*999
        #print((one[0],one[1]))
        q= [one]
        euc_dist=[]
        poped = []

        while (len(q)):
            poped = q[0]
            q.pop(0)

            # coordinate of currently popped node
            x = poped[0]
            y = poped[1]
            
            #print(poped, end ='')
       
            # now check for all adjancent of popped element
            for i in range(4):
                
                adjx = x + newx[i]
                adjy = y + newy[i]
                #print([x,y], end='')
                if (adjx >= 0 and adjx < m and adjy >= 0 and
                    adjy < n):
                    a = 1
                    e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                    if (visit[adjx][adjy] != 1):
                        q.append([adjx,adjy])
                        visit[adjx][adjy] = 1
                        #if mat[adjx][adjy]==0:
                        euc_dist.append(e)
                            #print("fg")
    #print(euc_dist)
    #print(visit)

                    #elif visit[adjx][adjy] < visit[x][y] +1:
                     #   visit[adjx][adjy] = visit[x][y] +1
                        # == 0:# visit[adjx][adjy] + 1:
                        #visit[adjx][adjy] = 1
                        #print((adjx,adjy),end='')
                    
                        #if mat[adjx][adjy]==0:
                         #   e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                          #  euc_dist.append(e)
                           # print("zero at"+str((adjx,adjy)))
                            #print(min(euc_dist)) 
                            #visit[adjy][adjy] = min(euc_dist)
                        #else:
                        #q.append([adjx,adjy])
                     #   visit[adjy][adjy] = 1
                    #else:
                       # q.append([adjx, adjy])
        euc_array[one[0]][one[1]] =  min(euc_dist)
        #print(visit)
    print(visit)

    #return euc_array
    
#%%
#Ploting

a = mdup_data
#a = lbt_data[23][:][:]
#a1 = best(a)
#a2 = pre_built(mat)
#a = easy3d
my_alg_time = time.time()
a1 = pre_built(a)
print("pre_build takes %s seconds " % (time.time() - my_alg_time))

#print(a1[15])
pre_alg_time = time.time()
a2 = best(a)
#a2 = pre_built(a)
print("mine takes %s seconds " % (time.time() - pre_alg_time))

#pre_alg_time = time.time()
#a3 = dans(a)
#a2 = pre_built(a)
#print("dans takes %s seconds " % (time.time() - pre_alg_time))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(a1)
plt.subplot(1,2,2)
plt.imshow(pre_built(a2))
#plt.subplot(1,3,3)
#plt.imshow(a3)
plt.show()

#def best(mat):
# %%


def best(mat):
    """editing"""
    #mat = mdup_data
    m = mat.shape[0]
    n = mat.shape[1]
    newx = [-1, 0, 1, 0]#1, 1,-1,-1]
    newy = [0, -1, 0, 1]#1,-1,-1, 1]
     
    ones=[]
    
    for i in range(m):
        for j in range(n): 
            if mat[i][j]==1:
                
                ones.append(np.array([i,j]))
            visit= np.ones(shape=(mat.shape))
            visit[one[0]][one[1]] = 0
            #euc_array[one[0]][one[1]] = 0
            visit = visit*999
            #euc_array = euc_array*999
            #print((one[0],one[1]))
            q= [one]
            euc_dist=[]
            poped = []

            while (len(q)):
                poped = q[0]
                q.pop(0)

                # coordinate of currently popped node
                x = poped[0]
                y = poped[1]
                
                #print(poped, end ='')
        
                # now check for all adjancent of popped element
                for i in range(4):
                    
                    adjx = x + newx[i]
                    adjy = y + newy[i]
                    #print([x,y], end='')
                    if (adjx >= 0 and adjx < m and adjy >= 0 and
                        adjy < n):
                        a = 1
                        e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                        if mat[adjx][adjy]==0:
                            euc_dist.append(e)
                            
                        elif visit[adjx][adjy] > visit[x][y] +1:
                            visit[adjx][adjy] = visit[x][y] +1
                            # == 0:# visit[adjx][adjy] + 1:
                            #visit[adjx][adjy] = 1
                            #print((adjx,adjy),end='')
                        
                            #if mat[adjx][adjy]==0:
                            #   e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                            #  euc_dist.append(e)
                            # print("zero at"+str((adjx,adjy)))
                                #print(min(euc_dist)) 
                                #visit[adjy][adjy] = min(euc_dist)
                            #else:
                            q.append([adjx,adjy])
                        #   visit[adjy][adjy] = 1
                        #else:
                        # q.append([adjx, adjy])
            euc_array[one[0]][one[1]] =  min(euc_dist)
            #print(visit)
        
        return euc_array
#%%

    #print(easy)
    
    #ones = [np.array([2,2]), np.array([0,3])]
    euc_array= np.zeros(shape=(mat.shape))
    #print(ones)

    for one in ones:
        visit= np.ones(shape=(mat.shape))
        visit[one[0]][one[1]] = 0
        #euc_array[one[0]][one[1]] = 0
        visit = visit*999
        #euc_array = euc_array*999
        #print((one[0],one[1]))
        q= [one]
        euc_dist=[]
        poped = []

        while (len(q)):
            poped = q[0]
            q.pop(0)

            # coordinate of currently popped node
            x = poped[0]
            y = poped[1]
            
            #print(poped, end ='')
       
            # now check for all adjancent of popped element
            for i in range(4):
                
                adjx = x + newx[i]
                adjy = y + newy[i]
                #print([x,y], end='')
                if (adjx >= 0 and adjx < m and adjy >= 0 and
                    adjy < n):
                    a = 1
                    e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                    if mat[adjx][adjy]==0:
                        euc_dist.append(e)
                        
                    elif visit[adjx][adjy] > visit[x][y] +1:
                        visit[adjx][adjy] = visit[x][y] +1
                        # == 0:# visit[adjx][adjy] + 1:
                        #visit[adjx][adjy] = 1
                        #print((adjx,adjy),end='')
                    
                        #if mat[adjx][adjy]==0:
                         #   e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                          #  euc_dist.append(e)
                           # print("zero at"+str((adjx,adjy)))
                            #print(min(euc_dist)) 
                            #visit[adjy][adjy] = min(euc_dist)
                        #else:
                        q.append([adjx,adjy])
                     #   visit[adjy][adjy] = 1
                    #else:
                       # q.append([adjx, adjy])
        euc_array[one[0]][one[1]] =  min(euc_dist)
        #print(visit)
    
    return euc_array
#%%
def best(mat):
    """editing"""
    #mat = mdup_data
    m = mat.shape[0]
    n = mat.shape[1]
    newx = [-1, 0, 1, 0]#1, 1,-1,-1]
    newy = [0, -1, 0, 1]#1,-1,-1, 1]
    visit= np.ones(shape=(mat.shape))
    euc_array = np.ones(shape=(mat.shape))
    visit = visit*999
    ones=[]
    zeros=[]
    for i in range(m):
        for j in range(n): 
            if mat[i][j]==1:
                ones.append(np.array([i,j]))
            else: 
                visit[i][j] = 0
                zeros.append(np.array([i,j]))
                #q.append([i,j])
                
    #print(easy)
    #ones = [np.array([2,2]), np.array([0,3])]
    euc_array= np.zeros(shape=(mat.shape))
    #print(ones)
     
    for one in ones:
        
        #visit[one[0]][one[1]] = 1
        #euc_array[one[0]][one[1]] = 0
        #euc_array = euc_array*999
        #print((one[0],one[1]))
        q= [one]
        euc_dist=[]
        poped = []

        while (len(q)):
            poped = q[0]
            q.pop(0)

            # coordinate of currently popped node
            x = poped[0]
            y = poped[1]
            
            #print(poped, end ='')
       
            # now check for all adjancent of popped element
            for i in range(4):
                
                adjx = x + newx[i]
                adjy = y + newy[i]
                #print([x,y], end='')
                if (adjx >= 0 and adjx < m and adjy >= 0 and
                    adjy < n):
                    eold  = np.sqrt((one[0]-x)**2+(one[1]-y)**2)
                    e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)

                    if e>eold:
                        q.append([adjx,adjy])
                    else:
                        euc_array[]
                    #if mat[adjx][adjy]==0:
                     #   euc_dist.append(e)
                        
                    #if visit[adjx][adjy] == visit[x][y] +1:
                        visit[adjx][adjy] = visit[x][y] +1
                        # == 0:# visit[adjx][adjy] + 1:
                        #visit[adjx][adjy] = 1
                        #print((adjx,adjy),end='')
                    
                        #if mat[adjx][adjy]==0:
                         #   e = np.sqrt((one[0]-adjx)**2+(one[1]-adjy)**2)
                          #  euc_dist.append(e)
                           # print("zero at"+str((adjx,adjy)))
                            #print(min(euc_dist)) 
                            #visit[adjy][adjy] = min(euc_dist)
                        #else:
                        q.append([adjx,adjy])
                     #   visit[adjy][adjy] = 1
                    #else:
                       # q.append([adjx, adjy])
        euc_array[one[0]][one[1]] =  min(euc_dist)
        #print(visit)
    #print(visit)

    return euc_array

#%%

a = mdup_data
a = lbt_data[15]
#a1 = best(mat)
#a2 = pre_built(mat)
#a = easy3d
my_alg_time = time.time()
a1 = pre_built(a)
print("pre_build takes %s seconds " % (time.time() - my_alg_time))

#print(a1[15])
pre_alg_time = time.time()
a2 = best(a)
#a2 = pre_built(a)
print("mine takes %s seconds " % (time.time() - pre_alg_time))

#pre_alg_time = time.time()
##a3 = dans(a)
#a2 = pre_built(a)
#print("dans takes %s seconds " % (time.time() - pre_alg_time))


plt.figure()
plt.subplot(1,3,1)
plt.imshow(a1)
plt.subplot(1,3,2)
plt.imshow(a2)
#plt.subplot(1,3,3)
#plt.imshow(a3)
plt.show()

# %%

# %%

def chamfer_distance3d(img):
   w, h,l = img.shape
   dt = np.zeros((w,h), np.uint32)
   # Forward pass
   x = 0
   y = 0
   z = 0
   if img[x,y,z] == 1:
      dt[x,y,z] = 65535 # some large value
   for x in range(1, w):
      if img[x,y,z] == 1:
         dt[x,y,z] = 3 + dt[x-1,y,z]
   for y in range(1, h):
      x = 0
      if img[x,y,z] == 1:
         dt[x,y,z] = min(3 + dt[x,y-1,z], 4 + dt[x+1,y-1,z])
      for x in range(1, w-1):
         if img[x,y] == 1:
            dt[x,y] = min(4 + dt[x-1,y-1], 3 + dt[x,y-1], 4 + dt[x+1,y-1], 3 + dt[x-1,y])
      x = w-1
      if img[x,y] == 1:
         dt[x,y] = min(4 + dt[x-1,y-1], 3 + dt[x,y-1], 3 + dt[x-1,y])
   # Backward pass
   for x in range(w-2, -1, -1):
      y = h-1
      if img[x,y] == 1:
         dt[x,y] = min(dt[x,y], 3 + dt[x+1,y])
   for y in range(h-2, -1, -1):
      x = w-1
      if img[x,y] == 1:
         dt[x,y] = min(dt[x,y], 3 + dt[x,y+1], 4 + dt[x-1,y+1])
      for x in range(1, w-1):
         if img[x,y] == 1:
            dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 4 + dt[x-1,y+1], 3 + dt[x+1,y])
      x = 0
      if img[x,y] == 1:
         dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 3 + dt[x+1,y])
   return dt



# %%
#a = easy
#a = mdup_data
a = lbt_data[15]
my_alg_time = time.time()
a1 = pre_built(a)
print("pre_build takes %s seconds " % (time.time() - my_alg_time))

#print(a1[15])
pre_alg_time = time.time()
a2 = chamfer_distance(a)
#a2 = pre_built(a)
print("new takes %s seconds " % (time.time() - pre_alg_time))

#pre_alg_time = time.time()
#a3 = best(a)
#a2 = pre_built(a)
#print("best takes %s seconds " % (time.time() - pre_alg_time))
#print(a1)
#print(a2)

#print(np.array_equal (a1, a2, equal_nan=False))
#print(np.allclose(a1,a2, rtol=.2, atol=.2))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(a2)
plt.subplot(1,3,2)
plt.imshow(pre_built(lbt_data)[15])
#plt.subplot(1,3,3)
#plt.imshow(a3)
plt.show()
#%%

# %%
def compute(x, axes=None, f=L2):
  """Compute the distance transform of a sampled function
  Compute the N-dimensional distance transform using the method described in:
    P. Felzenszwalb, D. Huttenlocher "Distance Transforms of Sampled Functions"
  Args:
    x (ndarray): An n-dimensional array representing the data term
  Keyword Args:
    axes (tuple): The axes over which to perform the distance transforms. The
      order does not matter. (default all axes)
    f (DistanceFunction): The distance function to apply (default L2)
  """
  shape = x.shape
  axes = axes if axes else tuple(range(x.ndim))
  f = f() if isinstance(f, type) else f

  # initialize the minima and argument arrays
  min = x.copy()
  arg = tuple(np.empty(shape, dtype=int) for axis in axes)

  # create some scratch space for the transforms
  v = np.empty((max(shape)+1,), dtype=int)
  z = np.empty((max(shape)+1,), dtype=float)

  # compute transforms over the given axes
  for n, axis in enumerate(axes):

    numel  = shape[axis]
    minbuf = np.empty((numel,), dtype=float)
    argbuf = np.empty((numel,), dtype=int)
    slices = map(xrange, shape)
    slices[axis] = [Ellipsis]

    for index in itertools.product(*slices):

      # compute the optimal minima
      _compute1d(min[index], f, minbuf, argbuf, z, v)
      min[index] = minbuf
      arg[n][index] = argbuf
      nindex = tuple(argbuf if i is Ellipsis else i for i in index)

      # update the optimal arguments across preceding axes
      for m in reversed(range(n)):
        arg[m][index] = arg[m][nindex]

  # return the minimum and the argument
  return min, arg

print(compute(a))
# %%
for y in np.arange(0,5,1):
        for x in np.arange(0,8,0.01):
            for z in np.arange(5,0,-0.01):
                #Collect the points in one array like in the following code.

                point = np.array([x,y,z])
                a1 = np.array([a.p1.x,a.p1.y,a.p1.z])
                a2 = np.array([a.p2.x,a.p2.y,a.p2.z])
                a3 = np.array([a.p3.x,a.p3.y,a.p3.z])

                if np.linalg.norm(point-a1) <=1:
                    print(point)
                    continue
                if np.linalg.norm(point-a2) <=1:
                    print(point)
                    continue
                if np.linalg.norm(point-a3) <=1:
                    print (point)
                    continue
#$$   
#It is better to store the points directly as numpy arrays in your object specifiedPoints[key] and not to collect them again and again in every loop. This would get you this code:

    point = np.array([x,y,z])

    if np.linalg.norm(point-a.p1) <=1:
        print point
        continue
    if np.linalg.norm(point-a.p2) <=1:
        print point
        continue
    if np.linalg.norm(point-a.p3) <=1:
        print point
        continue
# %%
