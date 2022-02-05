"""Created task3 task file to try commit changes"""
#%%
import numpy as np
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
        [0, 0, 0,1,0],
        [0,0,0,1,1],
        [0,0,1,1,1]]])

#%%
#dims =np.array([1,1,1])
array = easy3d[1]
image = easy3d[1]
array = (array - np.min(array))/(np.max(array) - np.min(array))
XX, YY = np.meshgrid(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), indexing='ij')
coord = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1)), axis =1)
ipts = np.arange(0, array.shape[0])
jpts = np.arange(0, array.shape[1])
points = (ipts, jpts)
values = array
xi = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1)),  axis =1)

image_interpn_flatten = interpn(points, array, xi)
image_interpn = image_interpn_flatten.reshape(image.shape)
print(xi.shape)
print(image_interpn_flatten.shape)
print(image_interpn.shape)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(image_interpn, cmap='gray',origin='lower', vmin=0, vmax=1)
ax[0].title.set_text('Original')
ax[1].title.set_text('Interpolated')
fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
plt.show()

print("img")
#%%
dims =np.array([1,1,1])
array = easy3d
image = easy3d
array = (array - np.min(array))/(np.max(array) - np.min(array))
#precomputing
#specify local image coordinates
#Easiest way is to use array indexes as coordinates instead of taking the 
# center of the voxel, which will be done during interpolation
XX, YY, ZZ = np.meshgrid(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), np.arange(0, array.shape[2]), indexing='ij')
#XX, YY = np.meshgrid(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), indexing='ij')

XX = XX*dims[0]
YY= YY*dims[1]
ZZ = ZZ*dims[2]
coord = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1), ZZ.reshape(-1,1)), axis =1)
#print(coord)

#self = 1
#array = easy3d
ipts = np.arange(0, array.shape[0])
jpts = np.arange(0, array.shape[1])
kpts = np.arange(0, array.shape[2])
points = (ipts, jpts, kpts)
values = array
xi = np.concatenate((XX.reshape(-1,1), YY.reshape(-1,1),  ZZ.reshape(-1,1)),  axis =1)

image_interpn_flatten = interpn(points, image, xi)
image_interpn = image_interpn_flatten.reshape(image.shape)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image[1], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(image_interpn[1], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[0].title.set_text('Original')
ax[1].title.set_text('Interpolated')
fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
plt.show()


#%%
class Image3D: 
    def __init__(self, array, dims = np.array([1,1,1])):
        #self.array = array
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

trying = Image3D(easy3d[1])
trans0 = np.array([[1,          0,                  1],
                  [0,           1,                  1],
                  [0,           0,                  1] ])

trying.warp3d(trans0)

#%%
class AffineTransform: 
    def __init__(self,tranform_parameters):
        self.tranform = tranform_parameters
        if tranform_parameters.shape != (3,3,3):
            print("transform shape incorrect")
        # check length of vector allowing 6/7 DoF for rigid
        # 12 dof for affine
        # should also allow non and generate random affine generation

        # precomute the transformation matric in homogenours coordinates
        # using rigid_transform or affine_transform 
        # and save it in a class member variable
        print("img")

    def random_transform_generator(img, s1 = None, s2 =None):
        if s1 == None: 
            s1 = 1 + 0.1*(np.random.randn(1))
        if s2 == None:
            s2 = 1 + 0.1*(np.random.randn(1))
        
        b = 0.1*(np.random.randn(1))
        d = 0.1*(np.random.randn(1))
        c = 10*(np.random.randn(1))
        f = 10*(np.random.randn(1))
        rand = np.array([[s1,b,c],[d,s2,f],[0,0,1]])
        return rand

        # returns random affine matrix in homogenous coordinates

        # cosidering
        # 1. design and implement s reasonable method for generating 
        #       random affine transformation of a 3d image
        # with a customisable scalar parameter to control 
        # the strength of the transformation.
        

    def rigid_transform(img):
        # 6/7 dof
        # taking the transformation parameter to return the 
        # transformation matrix in homogeneous coordinates.
        print("img")

    def affine_transform(img):
        # 12 dof
        # taking the transformation parameter to return the 
        # transformation matrix in homogeneous coordinates.
        print("img")

##
#%%
import numpy as np
# load image_train00.npy
# instatiate image 3d object

# manually define 10 rigid and affine transformations
# that demonstate a variety of rotation, translation, 
# scaling, general affine and combinations
tx = 40 #movement in x direction
ty = -40 #movement in y direction
x = 45 #angle
s = 0.5 #scaling

#translation
trans0 = np.array([[1,          0,                  tx],
                  [0,           1,                  ty],
                  [0,           0,                  1] ])

#rotation
trans1 = np.array([[np.cos(x),    (-1)(np.sin(x)),  0],
                  [np.sin(x),   np.cos(x),          0],
                  [0,           0,                  1] ])

trans2 = np.array([[np.cos(x),  0,    (-1)(np.sin(x))],
                  [0,           1,                  0],
                  [np.sin(x),   0,          np.cos(x)] ])
#rotation+translation
trans3 = np.array([[np.cos(x),    (-1)(np.sin(x)),  tx],
                  [np.sin(x),   np.cos(x),          ty],
                  [0,           0,                  1] ])
#affine
scal = np.array([[s,0,0],[0,s,0],[0,0,1]])
trans4 = scal*trans3

a = 1 + 0.1*(1.16748322)
b = 0.1*(-0.70569)
d = 0.1*(.281311)
e = 1 + 0.1*(.8965)
c = 10*1.532
f = 10*.606
trans5 = np.array([[a,b,c],[d,e,f],[0,0,1]])
trans6 = np.array([[e,d,c],[a,b,f],[0,0,1]])

tx = -5
ty = 10
a = 1 + 0.1*(0.1878)
b = 0.1*(-0.5245)
d = 0.1*(.20255)
e = 1 + 0.1*(-0.08786689)
c = 10*1.08533
f = 10*(-0.82893)

trans7 = np.array([[1,b,0],[d,1,0],[0,0,1]])
trans8 = np.array([[e,d,f],[a,b,c],[0,0,1]])
trans9= np.array([[1,          0,                  0],
                [0,           1,                  0],
                [tx,           ty,                  1] ])

#%%
# generate the warped images using these transformations

# generate 10 different randomaly warped images and plot
# 5 image slices for each transformed image at different 
# z depths

# change the strength parameter in random_transform_generator
# generate images with 5 different values for the strength 
# parameter. visualise the randomaly transformed images



# %%
