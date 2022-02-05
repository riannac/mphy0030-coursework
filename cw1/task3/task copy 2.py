"""Created task3 task file to try commit changes"""
#%%
import random
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
    
    
    def centering(self, affineTransform_input):
        image = self.array
        center = compute_center(image.shape)
        centering_tf = np.eye(4)
        centering_tf[0, 3] = 0 - center[0]
        centering_tf[1, 3] = 0 - center[1]
        centering_tf[2,3]=0-center[2]

        # inverse centering transformation
        centering_tf_inv = np.eye(4)
        centering_tf_inv[0, 3] = -centering_tf[0, 3]
        centering_tf_inv[1, 3] = -centering_tf[1, 3]
        centering_tf_inv[2,3]=-centering_tf[2,3]
        M = np.dot(centering_tf_inv, np.dot(affineTransform_input, centering_tf))
        return M

    def warp3d(self, affineTransform_input):
        # computes a warped 3d image with all voxel intensities 
        # interpolated by trilinear interpolation method
        # interpolation parameters
        image = self.array
        center = compute_center(image.shape)
        centering_tf = np.eye(4)
        centering_tf[0, 3] = 0 - center[0]
        centering_tf[1, 3] = 0 - center[1]
        centering_tf[2,3]=0-center[2]

        # inverse centering transformation
        centering_tf_inv = np.eye(4)
        centering_tf_inv[0, 3] = -centering_tf[0, 3]
        centering_tf_inv[1, 3] = -centering_tf[1, 3]
        centering_tf_inv[2,3]=-centering_tf[2,3]
        M = np.dot(centering_tf_inv, np.dot(affineTransform_input, centering_tf))
        
        ipts = np.arange(0, self.array.shape[0])
        jpts = np.arange(0, self.array.shape[1])
        kpts = np.arange(0, self.array.shape[2])
        points = (ipts, jpts, kpts)
        values = self.array
        
        xi = np.dot(M, self.coord)[:3].T
        image_interpn_flatten = interpn(points, values, xi, bounds_error=False, fill_value=0)
        image_interpn = image_interpn_flatten.reshape(values.shape)
        
        self.warp = image_interpn
        return image_interpn
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
    def __init__(self,tranform_parameters= None):
        if tranform_parameters == None:
            tranform_matrix = random_transform_generator()
        self.tP = tranform_parameters
        self.tM = tranform_matrix
        lenM = (len(self.tP))
        if lenM != 6 or lenM != 7 or lenM != 12:
            print("Transform Matrix Shape Incorrect")
        
        # check length of vector allowing 6/7 DoF for rigid
        # 12 dof for affine
        # should also allow non and generate random affine generation

        # precomute the transformation matric in homogenours coordinates
        # using rigid_transform or affine_transform 
        # and save it in a class member variable
        
    def random_transform_generator(s1 = None, s2 =None, s3=None):
        if s1 == None: 
            s1 = 1 + 0.1*(np.random.randn(1))
        if s2 == None:
            s2 = 1 + 0.1*(np.random.randn(1))
        if s3 == None:
            s3 = 1 + 0.1*(np.random.randn(1))
        
        b = 0.1*(-0.70569)
        c = 0.1*(.281311)
        g= 0.1*(.7654)
        e=0.1*(.333)
        i= 0.1*(.1234)
        j=0.1*.3456

        d = 10*1.532
        h = 10*.606
        l = 10*.5679

        rand = np.array([
            [s1,b,c,d],
            [e,s1,g,h],
            [i,j,s3,l],
            [0,0,0,1]])
        return rand

        # returns random affine matrix in homogenous coordinates

        # cosidering
        # 1. design and implement s reasonable method for generating 
        #       random affine transformation of a 3d image
        # with a customisable scalar parameter to control 
        # the strength of the transformation.
        

    def rigid_transform(transformation_parameters_vector):
        """return the rigit transformation matrix for vector parameters organised
        as (scaling(1), translation(3), rotation(3))"""
        # 6/7 dof
        # 6:    t(v) = Rv  + t 
        # 7:    t(v) = sRv + t
        tp = transformation_parameters_vector
        adj = 0 #adjust
        if len(tp) != 7:
            s = 1
        else:
            s = tp[0]
            adj = 1

        tx = tp[0+adj]
        ty = tp[1+adj]
        tz=tp[2+adj]
        q=tp[3+adj] 
        r=tp[4+adj] 
        t = tp[5+adj]
        
        # taking the transformation parameter to return the 
        # transformation matrix in homogeneous coordinates.
        rs = np.array([
                [1,0,0,0],
                [0,np.cos(q),np.sin(q),0],
                [0,np.cos(q),(-1)*np.sin(q),0],
                [0,0,0,1]
            ])
        ry = np.array([
                [np.cos(r),np.sin(r),0,0],
                [np.cos(r),(-1)*np.sin(r),0,0],
                [0,0,1,0],
                [0,0,0,1]
            ])
        rz = np.array([
                [np.cos(t),0,(-1)*np.sin(t),0],
                [0,1,0,0],
                [np.sin(t),0,np.cos(t),0,],
                [0,0,0,1]
            ])
        trans = np.array([ 
                [1,          0,      0,      tx],
                [0,          1,      0,      ty],
                [0,          0,      1,      tz],
                [0,          0,      0,      1]])

        r = np.dot(np.dot(rs,ry),rz)
        rig_trans = s*r + trans
        return rig_trans
        
    def affine_transform(transformation_parameters_vector):
        # 12:    t(v) = Av + t
        tp = transformation_parameters_vector
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
tx = 40 #movement in x direction
ty = -40 #movement in y direction
tz = 10
#x = 45 #angle
#s = 0.5 #scaling

#translation
def translation(tx=1,ty=1,tz=1):    
    trans0 = np.array([ 
        [1,          0,      0,      tx],
        [0,          1,      0,      ty],
        [0,          0,      1,      tz],
        [0,          0,      0,      1]])
    return(trans0)
#rotation
def compute_center(shape):
    return tuple([s // 2 for s in shape])

def rot(image, t=45):
    trans0 = np.array([ [1,          0,      0,      0],
                    [0,          np.cos(t),      -1*np.sin(t),      0],
                    [0,          np.sin(t),      np.cos(t),      0],
                    [0,          0,      0,      1]])
    # centering transformation
    center = compute_center(image.shape)
    centering_tf = np.eye(4)
    centering_tf[0, 3] = 0 - center[0]
    centering_tf[1, 3] = 0 - center[1]
    centering_tf[2,3]=0-center[2]

    # inverse centering transformation
    centering_tf_inv = np.eye(4)
    centering_tf_inv[0, 3] = -centering_tf[0, 3]
    centering_tf_inv[1, 3] = -centering_tf[1, 3]
    centering_tf_inv[2,3]=-centering_tf[2,3]
    M = np.dot(centering_tf_inv, np.dot(trans0, centering_tf))
    return M

def af(image, a=1,b=0,c=0,d=0,e=0,f=1,g=0,h=0,i=0,j=0,k=1,l=0):
    trans0 = np.array([
        [a,b,c,d],
        [e,f,g,h],
        [i,j,k,l],
        [0,0,0,1]])
    # centering transformation
    center = compute_center(image.shape)
    centering_tf = np.eye(4)
    centering_tf[0, 3] = 0 - center[0]
    centering_tf[1, 3] = 0 - center[1]
    centering_tf[2,3]=0-center[2]

    # inverse centering transformation
    centering_tf_inv = np.eye(4)
    centering_tf_inv[0, 3] = -centering_tf[0, 3]
    centering_tf_inv[1, 3] = -centering_tf[1, 3]
    centering_tf_inv[2,3]=-centering_tf[2,3]
    M = np.dot(centering_tf_inv, np.dot(trans0, centering_tf))
    return M

def scal(image,s=1):
    scal = np.array([[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,1]])
    return scal

image = mdup_data
#rotation+translation
trans0 = translation()
trans1 = translation(0,2,2)
trans2 = rot(image)
trans3 = rot(image,90)
trans4 = translation(0,1,1) + rot(image)
trans5 = scal(image,.5)*trans4

a = 1 + 0.1*(1.16748322)
f = 1 + 0.1*(.8965)
k= 1+0.1*(.98769)

b = 0.1*(-0.70569)
c = 0.1*(.281311)
g= 0.1*(.7654)
e=0.1*(.333)
i= 0.1*(.1234)
j=0.1*.3456

d = 10*1.532
h = 10*.606
l = 10*.5679

trans6 = af(image, a,b,c,d,e,f,g,h,i,j,k,l)
trans7 = af(image,f,e,i,l,b,k,j,d,c,j,a,h)

a = 1 + 0.1*(1.1782)
f = 1 + 0.1*(.9845)
k= 1+0.1*(.769)

b = 0.1*(-0.5679)
c = 0.1*(.1311)
g= 0.1*(.7498)
e=0.1*(.444)
i= 0.1*(.3412)
j=0.1*.4321

d = 10*1.456
h = 10*.806
l = 10*.59975

trans8 = af(image, a,b,c,d,e,f,g,h,i,j,k,l)
trans9 = af(image,f,e,i,l,b,k,j,d,c,j,a,h)


#%%
# generate the warped images using these transformations

# generate 10 different randomaly warped images and plot
# 5 image slices for each transformed image at different 
# z depths

# change the strength parameter in random_transform_generator
# generate images with 5 different values for the strength 
# parameter. visualise the randomaly transformed images



# %%
image = easy3d
image = mdup_data 
trying = Image3D(image)
M = trans4
i = trying.warp3d(M)
a = trying.array

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
