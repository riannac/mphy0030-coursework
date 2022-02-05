"""Created task3 task file to try commit changes"""
#%%
from hashlib import sha3_224
import random
from matplotlib import transforms
import numpy as np
from scipy import rand
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
#%%
mdup_data = np.array([
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1,1,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,1,1,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,1,1,0,0,0],
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
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,1,1,1,0,0],
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

easy3d = np.array([
        [[0, 0, 1, 0, 0],
        [ 0, 1, 1, 1, 0],
        [ 1, 1, 1, 1, 1],
        [ 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0]],

        [[0, 0, 1, 0, 0],
        [ 0, 0, 1, 1, 0],
        [ 0, 0, 1, 1, 1],
        [ 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0]],

        [[0, 0, 1, 0, 0],
        [ 0, 1, 1, 0, 0],
        [ 1, 1, 0, 0, 0],
        [ 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0]]])


#%%

#%%
class Image2D:
    """working for 2d""" 
    def __init__(self, array, dims = np.array([1,1,1])):
        self.array = (array - np.min(array))/(np.max(array) - np.min(array))
        #precomputing
        #specify local image coordinates
        #Easiest way is to use array indexes as coordinates instead of taking the 
        # center of the voxel, which will be done during interpolation
        #XX, YY, ZZ = np.meshgrid(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), np.arange(0, array.shape[2]), indexing='ij')
        XX, YY = np.meshgrid(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), indexing='ij')
        
        self.XX = XX#*dims[0]
        self.YY=YY#*dims[1]
        #self.ZZ = ZZ#*dims[2]
        #self.coord = np.concatenate((self.XX.reshape(-1,1), self.YY.reshape(-1,1), self.ZZ.reshape(-1,1)), axis=1)
        #self.coord = np.concatenate((self.XX.reshape(-1,1), self.YY.reshape(-1,1)), axis=1)
        self.coord = np.concatenate((self.XX.reshape(-1,1), self.YY.reshape(-1,1), np.ones((np.prod(self.array.shape), 1))), axis=1).T
        
    def warp3d(self, affineTransform_input):
        # computes a warped 3d image with all voxel intensities 
        # interpolated by trilinear interpolation method
        # interpolation parameters
        ipts = np.arange(0, self.array.shape[0])
        jpts = np.arange(0, self.array.shape[1])
        #kpts = np.arange(0, self.array.shape[2])
        points = (ipts, jpts)#, kpts)
        
        #points = (ipts, jpts, kpts)
        values = self.array
        
        #xi = np.concatenate((self.XX.reshape(-1,1), self.YY.reshape(-1,1), self.ZZ.reshape(-1,1)), axis=1)
        xi = np.dot(affineTransform_input, self.coord)[:2].T
        image_interpn_flatten = interpn(points, values, xi, bounds_error=False, fill_value=0)
        image_interpn = image_interpn_flatten.reshape(values.shape)

        # interpolation
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(values, cmap='gray',origin='lower', vmin=0, vmax=1)
        ax[1].imshow(image_interpn, cmap='gray',origin='lower', vmin=0, vmax=1)
        ax[0].title.set_text('Original')
        ax[1].title.set_text('Interpolated')
        fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
        plt.show()
        self.warp = image_interpn


class Image3D:
    """doing something for 3d""" 
    def __init__(self, array, dims = np.array([1,1,1])):
        self.array = (array - np.min(array))/(np.max(array) - np.min(array))
        #precomputing
        #specify local image coordinates
        #Easiest way is to use array indexes as coordinates instead of taking the 
        # center of the voxel, which will be done during interpolation
        XX, YY, ZZ = np.meshgrid(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), np.arange(0, array.shape[2]), indexing='ij')
        
        self.XX = XX*dims[0]
        self.YY=YY*dims[1]
        self.ZZ = ZZ*dims[2]
        self.coord = np.concatenate((self.XX.reshape(-1,1), self.YY.reshape(-1,1), self.ZZ.reshape(-1,1), np.ones((np.prod(self.array.shape), 1))), axis=1).T
    
    def compute_center(shape):
            return tuple([s // 2 for s in shape])

    def warp3d(self, affineTransform_input):
        # computes a warped 3d image with all voxel intensities 
        # interpolated by trilinear interpolation method
        # interpolation parameters
        image = self.array
        
        # centering transform 
        #tuple([s//2 for s in image.shape])
        
        center = self.compute_center((image.shape))
        #tuple([s // 2 for s in image.shape])
        centering_tf = np.eye(4)
        centering_tf[0, 3] = 0 - center[0]
        centering_tf[1, 3] = 0 - center[1]
        centering_tf[2, 3] = 0 - center[2]

        # inverse centering transformation
        centering_tf_inv = np.eye(4)
        centering_tf_inv[0, 3] = -centering_tf[0, 3]
        centering_tf_inv[1, 3] = -centering_tf[1, 3]
        centering_tf_inv[2, 3] = -centering_tf[2,3]
        M = np.dot(centering_tf_inv, np.dot(affineTransform_input, centering_tf))
        
        ipts = np.arange(0, self.array.shape[0])
        jpts = np.arange(0, self.array.shape[1])
        kpts = np.arange(0, self.array.shape[2])
        points = (ipts, jpts, kpts)
        
        xi = np.dot(M, self.coord)[:3].T
        image_interpn_flatten = interpn(points, image, xi, bounds_error=False, fill_value=0)
        image_interpn = image_interpn_flatten.reshape(image.shape)
        
        self.warp = image_interpn
        return image_interpn
#%%
"""
transform = AffineTransform().rigid_transform()
 
elif lenM == 6 or lenM == 7:
    print(str(lenM) + " Transform parameters.")
    print("Should be in order, Scaling, translation, rotation, afine")
    return rigid_transform(self.tP)
elif lenM == 12:
    print(str(lenM) + " Transform parameters.")
    print("Should be in order, Scaling, translation, rotation, afine")
    return affine_transform(self.tP)
"""

#%%
class AffineTransform: 
    def __init__(self,tranform_parameters):
        #if tranform_parameters == None:
        #    return self.random_transform_generator()
        self.tP = tranform_parameters
        #self.tM = tranform_matrix
        lenM = (len(self.tP))
        if lenM != 6 and lenM != 7 and lenM != 12:
            print("Transform Matrix Shape Incorrect")
        
        # check length of vector allowing 6/7 DoF for rigid
        # 12 dof for affine
        # should also allow non and generate random affine generation

        # precomute the transformation matric in homogenours coordinates
        # using rigid_transform or affine_transform 
        # and save it in a class member variable
        
    def random_transform_generator(s1 = 0, s2 =0, s3=0):
        if s1 == 0: 
            s1 = 1 + 0.1*(np.random.randn(1))
        if s2 == 0:
            s2 = 1 + 0.1*(np.random.randn(1))
        if s3 == 0:
            s3 = 1 + 0.1*(np.random.randn(1))
        
        b = 0.1*(-0.70569)
        c = 0.1*(.281311)
        g= 0.1*(.7654)
        e= 0.1*(.333)
        i= 0.1*(.1234)
        j= 0.1*(.3456)

        d = 10*(1.532)
        h = 10*(.606)
        l = 10*(.5679)

        rand = np.array([
            [s1[0],b,c,d],
            [e,s2[0],g,h],
            [i,j,s3[0],l],
            [0,0,0,1]])
        return rand

        # returns random affine matrix in homogenous coordinates

        # cosidering
        # 1. design and implement s reasonable method for generating 
        #       random affine transformation of a 3d image
        # with a customisable scalar parameter to control 
        # the strength of the transformation.
        

    def rigid_transform(self):
        """return the rigit transformation matrix for vector parameters organised
        as (scaling(1), translation(3), rotation(3))"""
        # 6/7 dof
        # 6:    t(v) = Rv  + t 
        # 7:    t(v) = sRv + t
        tp = self.tP
        adj = 0 #adjust
        if len(tp) != 7:
            s = 1
        else:
            s = tp[0]
            adj = 1

        tx = tp[0+adj]
        ty = tp[1+adj]
        tz = tp[2+adj]
        r = tp[3+adj] 
        t = tp[4+adj] 
        q = tp[5+adj]
        print(r)

        # taking the transformation parameter to return the 
        # transformation matrix in homogeneous coordinates.
        s = np.array([[s,0,0,0],
            [0,s,0,0],
            [0,0,s,0],
            [0,0,0,1]])
        rz = np.array([
                [1,     0,      0,           0],
                [0,np.cos(q),(-1)*np.sin(q),0],
                [0,np.sin(q),np.cos(q),     0],
                [0,0,0,1]
            ])
        rx = np.array([
                [np.cos(r),(-1)*np.sin(r),0,0],
                [np.sin(r), np.cos(r),0,0],
                [0,0,1,0],
                [0,0,0,1]
            ])
        ry = np.array([
                [np.cos(t),0,np.sin(t),0],
                [0,1,0,0],
                [(-1)*np.sin(t),0,np.cos(t),0,],
                [0,0,0,1]
            ])
        trans = np.array([ 
                [0,          0,      0,      tx],
                [0,          0,      0,      ty],
                [0,          0,      0,      tz],
                [0,          0,      0,      0]])

        #r = np.dot(rz,np.dot(rs,ry))
        #r = rs*ry*rz
        rig_trans = np.dot(np.dot(np.dot(rx,ry),rz),s) + trans
        return rig_trans
        
    def affine_transform(self):
        # 12:    t(v) = Av + t
        tp = self.tP
        if len(tp) != 12:
            print("Not enough parameters for affine transformation")
        s1 = tp[0]
        b =tp[1]
        c =tp[2]
        tx = tp[3]

        e = tp[4]
        s2 = tp[5]
        g= tp[6]
        ty = tp[7]

        i = tp[8]
        j = tp[9]
        s3 = tp[10]
        tz=tp[11]

        rand = np.array([
            [s1,b,c,tx],
            [e,s2,g,ty],
            [i,j,s3,tz],
            [0,0,0,1]])
        # 12 dof
        # taking the transformation parameter to return the 
        # transformation matrix in homogeneous coordinates.
        return rand

##
#%%
# load image_train00.npy
# instatiate image 3d object

# manually define 10 rigid and affine transformations
# that demonstate a variety of rotation, translation, 
# scaling, general affine and combinations


#%%
# generate the warped images using these transformations

# generate 10 different randomaly warped images and plot
# 5 image slices for each transformed image at different 
# z depths

# change the strength parameter in random_transform_generator
# generate images with 5 different values for the strength 
# parameter. visualise the randomaly transformed images
t =[1,0,0,1,0,1,0,0,0,0,1,0]
print(t)
#print(len(t))
transformation = AffineTransform(t).affine_transform()
print(transformation)

#image = easy3d
image = mdup_data 
trying = Image3D(image)
M = transformation
a = trying.array
i = trying.warp3d(M)

#print(M)
#print(i)

# interpolation
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a[0], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(i[0], cmap='gray',origin='lower', vmin=0, vmax=1)     
ax[0].title.set_text('Original')
ax[1].title.set_text('Interpolated')
#fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a[1], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(i[1], cmap='gray',origin='lower', vmin=0, vmax=1)     
ax[0].title.set_text('Original')
ax[1].title.set_text('Interpolated')
#fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a[2], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(i[2], cmap='gray',origin='lower', vmin=0, vmax=1)     
ax[0].title.set_text('Original')
ax[1].title.set_text('Interpolated')
#fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))

plt.show()
# %%
M = trans1
p =(np.where((M!= 0) & (M!=1)))
print(trans0)
print(trans4)
print(trans5)
# %%
def compute_center(shape):
    return tuple([s // 2 for s in shape])

# %%
compute_center((2,5))
# %%
