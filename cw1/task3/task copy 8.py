"""Created task3 task file to try commit changes"""
#%%
from re import A
import numpy as np
from scipy import rand
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

import urllib.request
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
        #precomputing
        #Normalise array to values between 0 and 1 
        self.array = (array - np.min(array))/(np.max(array) - np.min(array))
        self.d = dims
        #self.shape = (array.shape[0]* dims[0], array.shape[1]*dims[1], array.shape[2]*dims[2])
        #self.shape = array.shape
        #specify local image coordinates
        # Easiest way is to use array indexes as coordinates instead of taking the 
        # center of the voxel, which will be done during interpolation
        x = np.arange(0, array.shape[0])
        y = np.arange(0, array.shape[1])
        z = np.arange(0, array.shape[2])
        
        XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
        
        self.XX = XX*dims[0]
        self.YY = YY*dims[1]
        self.ZZ = ZZ*dims[2]
        self.coord = np.concatenate((self.XX.reshape(-1,1), self.YY.reshape(-1,1), self.ZZ.reshape(-1,1), np.ones((np.prod(self.array.shape), 1))), axis=1).T
    
    def warp3d(self, affineTransform_input):
        # computes a warped 3d image with all voxel intensities 
        # interpolated by trilinear interpolation method
        
        image = self.array
        
        # transform matrix being centered
        #center = tuple([s //2 for s in self.shape])
        center = (image.shape[0]*self.d[0] /2, image.shape[1]*self.d[1]/2, image.shape[2]*self.d[2]/2)
        tf_matrix_centred = np.eye(4)
        tf_matrix_centred[0, 3] = 0 - center[0]
        tf_matrix_centred[1, 3] = 0 - center[1]
        tf_matrix_centred[2, 3] = 0 - center[2]

        # transform matrix being de-centered 
        tf_matrix_decentered = np.eye(4)
        tf_matrix_decentered[0, 3] = -tf_matrix_centred[0, 3]
        tf_matrix_decentered[1, 3] = -tf_matrix_centred[1, 3]
        tf_matrix_decentered[2, 3] = -tf_matrix_centred[2,3]
        MTM = np.dot(tf_matrix_decentered, np.dot(affineTransform_input, tf_matrix_centred))
        
        # interpolation parameters
        ipts = np.arange(0, self.array.shape[0])*self.d[0]
        jpts = np.arange(0, self.array.shape[1])*self.d[1]
        kpts = np.arange(0, self.array.shape[2])*self.d[2]
        #ipts = np.linspace(-self.array.shape[0] + 0.5, self.array.shape[0] - 0.5,array.shape[0])*self.d[0]
        #jpts = np.linspace(-self.array.shape[1] + 0.5, self.array.shape[1] - 0.5,array.shape[1])*self.d[0]
        #kpts = np.linspace(-self.array.shape[2] + 0.5, self.array.shape[2] - 0.5,array.shape[2])*self.d[0]
        self.points = (ipts, jpts, kpts)
        
        MTM_lcl_coords = np.dot(MTM, self.coord)[:3].T
        image_interpn_flatten = interpn(self.points, image, MTM_lcl_coords, bounds_error=False, fill_value=0)
        image_interpn = image_interpn_flatten.reshape(image.shape)
        self.warp = image_interpn

        return image_interpn


class AffineTransform: 
    def __init__(self, tranform_parameter, s =[0,0,0]):
        """Rigid transform should be in order, Scaling, Translation, Rotation. Default scaling is 1.
        Affine transform should be in order of matrix elements."""
        if tranform_parameter != None: 
            self.tP = tranform_parameter
            lenM = (len(self.tP))
        
            if lenM != 6 and lenM != 7 and lenM != 12:
                print("Error : Transform Matrix Shape Incorrect")
            elif lenM == 6:
                #print("Rigid Transform with " + str(lenM) + " transform parameters.")
                #print("Should be in order, Translation(3), Rotation(3)")
                self.tM = self.rigid_transform()
            elif lenM == 7:
                #print("Rigid Transform with " +str(lenM) + " Transform parameters.")
                #print("Should be in order, Scaling, Translation, Rotation")
                self.tM = self.rigid_transform()
            elif lenM == 12:
                #print("Rigid Transform with " +str(lenM) + " Transform parameters.")
                #print("Should be in order of Matrix elements")
                self.tM = self.affine_transform()
        else:
            self.s = s
            self.tM = self.random_transform_generator(s)

    # precomute the transformation matric in homogenours coordinates
    # using rigid_transform or affine_transform 
    # and save it in a class member variable
    rt = np.array([[-1,  0,  0,  0,],
                    [ 0, -1,  0,  0],
                    [ 0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  1.]])
    at = np.array([[ 2.1, -.705,  .2813,  15],
                    [ 3.3,  1.010,  .7654,  6.0],
                    [.12340,  3.4,  1.0267,  5.6790],
                    [ 0,  0,  0,  1.]])
    #random_transform_generator = random_transform_generator()
        
    def random_transform_generator(self,s =[0,0,0]):
        """Returns random affine matrix in homogenous coordinates. 
        Can set scaling, with list of x,y,z scale parameters. 
        If only 1 value is given, it will be used for all. 
        If two are given, it will be used for x,y and z will be random.
        setting s1,s2,s3 means you are dotting by scalling matrix, 
        so you'll generally get a non zero result"""
        if len(s) ==1:
            s1 = s[0]
            s2 = s[0]
            s3 = s[0]
        elif len(s) ==2:
            s1 = s[0]
            s2 = s[1]
            s3 = 1
        elif len(s)==3:
            if s[0] == 0: 
                s1 = 1
            else:
                s1 = s[0]

            if s[1] == 0:
                s2 = 1
            else:
                s2 = s[1]

            if s[2] == 0:
                s3 =1
            else: 
                s3=s[2]
        
        a =(s1) + 0.1*(np.random.randn(1))
        f =(s2) + 0.1*(np.random.randn(1))
        k =(s3) + 0.1*(np.random.randn(1))

        b= 0.1*(np.random.randn(1))
        c= 0.1*(np.random.randn(1))
        g= 0.1*(np.random.randn(1))
        e= 0.1*(np.random.randn(1))
        i= 0.1*(np.random.randn(1))
        j= 0.1*(np.random.randn(1))

        d = 10*(np.random.randn(1))
        h = 10*(np.random.randn(1))
        l = 10*(np.random.randn(1))

        rand = np.array([
            [a[0],b[0],c[0],d[0]],
            [e[0],f[0],g[0],h[0]],
            [i[0],j[0],k[0],l[0]],
            [0,0,0,1]])
        #return np.dot(rand,s)
        return(rand)
        
        # cosidering
        # 1. design and implement s reasonable method for generating 
        #       random affine transformation of a 3d image
        # with a customisable scalar parameter to control 
        # the strength of the transformation.
    def cos_ap(self,x, rnd = 6):
        co = round(np.cos(x),rnd)
        return co
    def sin_ap(self,x, rnd = 6):
        si = round(np.sin(x),rnd)
        return si


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
        

        # taking the transformation parameter to return the 
        # transformation matrix in homogeneous coordinates.
        s = np.array([[s,0,0,0],
            [0,s,0,0],
            [0,0,s,0],
            [0,0,0,0]])
        rz = np.array([
                [1,     0,      0,           0],
                [0,self.cos_ap(q),(-1)*self.sin_ap(q),0],
                [0,self.sin_ap(q),self.cos_ap(q),     0],
                [0,0,0,0]
            ])
        rx = np.array([
                [self.cos_ap(r),(-1)*self.sin_ap(r),0,0],
                [self.sin_ap(r), self.cos_ap(r),0,0],
                [0,0,1,0],
                [0,0,0,0]
            ])
        ry = np.array([
                [self.cos_ap(t),0,self.sin_ap(t),0],
                [0,1,0,0],
                [(-1)*self.sin_ap(t),0,self.cos_ap(t),0],
                [0,0,0,0]
            ])
        trans = np.array([ 
                [0,          0,      0,      tx],
                [0,          0,      0,      ty],
                [0,          0,      0,      tz],
                [0,          0,      0,      1]])

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


def plot_both(a,i,m):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(a[m], cmap='gray',origin='lower', vmin=0, vmax=1)
    ax[1].imshow(i[m], cmap='gray',origin='lower', vmin=0, vmax=1)     
    ax[0].title.set_text('Original')
    ax[1].title.set_text('Transformed')
    plt.show()
#%%

# manually define 10 rigid and affine transformations
t00 = [1,0,0,0,0,0,0]
t0 = [0,2.5,1,0,0,(np.pi)]
t1 = [.5, 0,0,1,.45,.57,0]
t2 = [.5,1,.5,1,(np.pi),(2*np.pi),0]

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

t3 = [1,-b,-c,0,b,1,-e,0,-c-.1,e+.5,1,0]
t4 = [1,0,0,5,0,1,1.3,5,0,1.3,1,5]
t5 = [1,0,0,5,.3,1,.5,15,0,0,1,5]

t6 = [a,b,c,d,e,f,g,h,i,j,k,l]
t7 = [f+.1,e+.05,i+.5,l+.2,b+.2,k+.05,j+.08,d+.56,c+.76,j+.22,a+.1,h+.111]

a = 1 + 0.1*(1.782)
f = 1 + 0.1*(.451111)
k= 1+0.1*(.12769)

b = 0.1*(0.025679)
c = 0.1*(.1311)
g= 0.1*(.7498)
e=0.1*(.444)
i= 0.1*(.3412)
j=0.1*.4321

d = 10*1.456
h = 10*.806
l = 10*.59975

t8 = [a+.12,b+.234,c+.0145,d+.787,e+.112,f+.11,g+.7,-h+.643,i+.198,-j+.44,k+.08,l+.8]
t9 = [f,e,i,l,b,k,j,d,c,j,a,h]



# load image_train00.npy
lbt_data = np.load('image_train00.npy',allow_pickle=False)
# instatiate image 3d object
trying = Image3D(lbt_data, np.array([.25,1,1]))
comp_image = trying.array
# manually define matrixes 
# that demonstate a variety of rotation, translation, 
# scaling, general affine and combinations
parameters = [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9]
trans = []
for para in parameters: 
    trans.append(AffineTransform(para).tM)

# generate the warped images using these transformations
images = []
for transformation in trans: 
    im = Image3D(lbt_data,np.array([2,.5,.5])).warp3d(transformation)
    images.append(im)

# generate 10 different randomaly warped images and plot
# 5 image slices for each transformed image at different 
# z depths

rand_t = []
rand_im = []

for ran in range(10): 
    r_t =AffineTransform(None).tM 
    rand_t.append(r_t)
    r_im = Image3D(lbt_data,np.array([2,.5,.5])).warp3d(r_t)
    rand_im.append(r_im)
#%%
a = comp_image

i = images[9]
m = 4
fig = plt.figure()
#fig.imshow(comp_image[m], cmap='gray',origin='lower', vmin=0, vmax=1)
plt.imshow(i[m], cmap='gray',origin='lower', vmin=0, vmax=1)     
plt.title('Transformed')
plt.show()
#%%
# change the strength parameter in random_transform_generator
# generate images with 5 different values for the strength 
# parameter. visualise the randomaly transformed images

r_scal = []
for scaling in [[2],[1,4,4],[1.5],[0.5],[1/10,1/10]]:
    r_t =AffineTransform(None,scaling).tM 
    r_im = Image3D(lbt_data,np.array([.25,1,1])).warp3d(r_t)
    r_scal.append(r_im)


#%%
#image = easy3d
#image = im
#trying = Image3D(image)
M = transformation
a = trying.array
i = trying.warp3d(M)
m = 5
n=6
v=7
# interpolation

def plot_both(a,i,m):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(a[m], cmap='gray',origin='lower', vmin=0, vmax=1)
    ax[1].imshow(i[m], cmap='gray',origin='lower', vmin=0, vmax=1)     
    ax[0].title.set_text('Original')
    ax[1].title.set_text('Transformed')
    plt.show()

def plot_trans(i,m):
    fig, ax = plt.figure()
    ax.imshow(a[m], cmap='gray',origin='lower', vmin=0, vmax=1)
    ax.title.set_text('Transformation')
    plt.show()


#fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a[n], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(i[n], cmap='gray',origin='lower', vmin=0, vmax=1)     
ax[0].title.set_text('Original')
ax[1].title.set_text('Transformed')
#fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a[v], cmap='gray',origin='lower', vmin=0, vmax=1)
ax[1].imshow(i[v], cmap='gray',origin='lower', vmin=0, vmax=1)     
ax[0].title.set_text('Original')
ax[1].title.set_text('Transformed')
#fig.suptitle('Warp of the original and interpolated images = ' + str(np.mean((image - image_interpn)**2)))
#plt.xlim([0,200])
plt.show()

# %%
print(trying.coord)
print(trying.points)
# %%
