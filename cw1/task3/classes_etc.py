import numpy as np
from scipy import rand
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import urllib.request
#%%

####################################################################################
class Image3D:
    """Warps images using a transformation matrix""" 
    def __init__(self, array, dims = np.array([1,1,1])):
        """ Normalise array and specify local image coordinates
            Inputs: Array = Image in the form of an array, 
                    Dims = Voxel dimentions in array format        """


        self.array = (array - np.min(array))/(np.max(array) - np.min(array))
        self.d = dims
        
        # Specify local image coordinates
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
        """Computes a warped 3d image with all voxel intensities 
         interpolated by trilinear interpolation method"""
        image = self.array
        
        # transform matrix being centered
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
        self.points = (ipts, jpts, kpts)
        
        MTM_lcl_coords = np.dot(MTM, self.coord)[:3].T
        image_interpn_flatten = interpn(self.points, image, MTM_lcl_coords, bounds_error=False, fill_value=0)
        image_interpn = image_interpn_flatten.reshape(image.shape)
        self.warp = image_interpn

        return image_interpn

#####################################################################################
class AffineTransform: 
    def __init__(self, tranform_parameter, st = 1):
        """Rigid transform with 7 DoF should be in order, Scaling(1), Translation(3), Rotation(3). 
        If 6 degrees of freedom, default scaling is 1.
        Affine transform should be in order of matrix elements."""

        if tranform_parameter != None: 
            self.tP = tranform_parameter
            lenM = (len(self.tP))
        
            if lenM != 6 and lenM != 7 and lenM != 12:
                self.tM="Error : Transform Matrix Shape Incorrect"
                print(self.tM)
                return

            elif lenM == 6:
                self.tM = self.rigid_transform()
            elif lenM == 7:
                self.tM = self.rigid_transform()
            elif lenM == 12:
                self.tM = self.affine_transform()
        else:
            self.st = st
            self.tM = self.random_transform_generator(st)

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
        
    def random_transform_generator(self,st = 1):
        """Returns random affine matrix in homogenous coordinates. 
        st = Strength parameter, controls how strong the transform. 
        if set to 0, it will return an identity matrix."""
        # I took scalar parameter to mean a number to scale the strength, not the scale of the image
        a =st*1*(np.random.randn(1))+1
        f =st*1*(np.random.randn(1))+1
        k =st*1*(np.random.randn(1))+1

        b =st*0.1*(np.random.randn(1))
        c =st*0.1*(np.random.randn(1))
        g =st*0.1*(np.random.randn(1))
        e =st*0.1*(np.random.randn(1))
        i =st*0.1*(np.random.randn(1))
        j =st*0.1*(np.random.randn(1))

        d =st*10*(np.random.randn(1))
        h =st*10*(np.random.randn(1))
        l =st*10*(np.random.randn(1))

        rand = np.array([
            [a[0],b[0],c[0],d[0]],
            [e[0],f[0],g[0],h[0]],
            [i[0],j[0],k[0],l[0]],
            [0,0,0,1]])

        return(rand)
        
    def cos_ap(self,x, rnd = 6):
        """returns cos(x) with rnd number of decimals"""
        co = round(np.cos(x),rnd)
        return co
    def sin_ap(self,x, rnd = 6):
        """returns sin(x) with rnd number of decimals"""
        si = round(np.sin(x),rnd)
        return si

    def rigid_transform(self):
        """returns the rigid transformation matrix, in homogeneous coordinates,
        for vector parameters organised as (scaling(1), translation(3), rotation(3))"""
        # 6/7 dof

        tp  = self.tP
        adj = 0                 #adjust
        if len(tp) != 6 and len(tp) != 7:
            print("Wrong number of parameters for rigid transformation")
            return

        if len(tp) != 7:
            s = 1
        else:
            s = 1/tp[0]
            adj = 1

        tx = tp[0+adj]
        ty = tp[1+adj]
        tz = tp[2+adj]
        r  = tp[3+adj] 
        t  = tp[4+adj] 
        q  = tp[5+adj]
        
        s = np.array([
            [s,0,0,0],
            [0,s,0,0],
            [0,0,s,0],
            [0,0,0,1]])
        rz = np.array([
            [1,     0,              0,                      0],
            [0,     self.cos_ap(q), (-1)*self.sin_ap(q),    0],
            [0,     self.sin_ap(q), self.cos_ap(q),         0],
            [0,     0,              0,                      1]
            ])
        rx = np.array([
            [self.cos_ap(r),    (-1)*self.sin_ap(r),    0,      0],
            [self.sin_ap(r),    self.cos_ap(r),         0,      0],
            [0,                 0,                      1,      0],
            [0,                 0,                      0,      1]
            ])
        ry = np.array([
            [self.cos_ap(t),        0,  self.sin_ap(t),     0],
            [0,                     1,  0,                  0],
            [(-1)*self.sin_ap(t),   0,  self.cos_ap(t),     0],
            [0,                     0,  0,                  1]
            ])
        trans = np.array([ 
            [1,          0,      0,      tx],
            [0,          1,      0,      ty],
            [0,          0,      1,      tz],
            [0,          0,      0,      1]])

        rig_trans = np.dot(np.dot(np.dot(np.dot(trans,rx), ry),rz),s) 
        return rig_trans
        
    def affine_transform(self):
        """return the affine transformation matrix, in homogeneous coordinates,
        for vector parameters,organised in order of matrix elements"""
        # 12 DoF

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
        
        return rand

# I've done the plotting in matplotlib, as the cw did not specify to save as png.
# but i have done that too.
def plot_both(a,i,m):
    """Plot and save comparison of transformed slice and original slice"""
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(a[m], cmap='gray',origin='lower', vmin=0, vmax=1)
    ax[1].imshow(i[m], cmap='gray',origin='lower', vmin=0, vmax=1)     
    ax[0].title.set_text('Original')
    ax[1].title.set_text('Transformed')
    plt.savefig('../task3/OG_n_trans_slice'+str(m)+'.png')
    plt.show()

def plot_slices(image_list, title='', slist = [3,9,17,22,25]):
    """Plot and save 5 slices of transformed image"""
    num = 0 
    for im in image_list:
        count = 0
        fig, ax = plt.subplots(1, 5)
        fig.set_size_inches(10,2.5) 
        for depth in slist:
            ax[count].imshow(im[depth], cmap='gray',origin='lower', vmin=0, vmax=1)
            ax[count].title.set_text('Depth = ' + str(depth))
            count += 1
        num += 1 
        plt.suptitle(title)
        plt.savefig('../task3/'+title+str(num)+'.png', transparent = False)
        plt.show()

from PIL import Image
import os


def save_im(array, title):
    """Saves an array as an image using pillow, with title input including image type, ex:.png """
    array = (array-array.min()) / (array.max()-array.min()) *255 
    im = Image.fromarray(array.astype('uint8'))
    im.save(title)

def pil_silce(images,slist =[3,9,17,22,25],t = "try"):
    c=0
    dir = "../task3/"+ t
    os.mkdir(dir)
    for image in images:
        t_i = t+str(c)
        c+=1
        #print(c)
        for slice in slist:
            array = image[slice]
            dem = array.max()-array.min()
            if dem == 0:
                array = (array- array.min())*0
            else:
                array = (array-array.min()) / (dem) *255
            #array = (array-array.min()) / (array.max()-array.min()) *255
            im = Image.fromarray(array.astype('uint8'))
            title = dir+"/"+t_i + "_s"+str(slice)+".png"
            #print(title)
            im.save(title)
#%%
#################################################################################
#################################################################################
# manually define 10 rigid and affine transformations
t00 = [1,0,0,0,0,0,0]
t0 = [0,20.5,10,0,0,(np.pi)]
t1 = [.8, 0,0,0,0,0,.57]
t2 = [1.7,1,5,5,(np.pi),(2*np.pi),0]

a = 1 + 0.1*(1.16748322)
f = 1 + 0.1*(.8965)
k = 1+0.1*(.98769)

b = 0.1*(-0.70569)
c = 0.1*(.281311)
g = 0.1*(.7654)
e =0.1*(.333)
i = 0.1*(.1234)
j =0.1*.3456

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

#################################################################################
#################################################################################
# load image_train00.npy


lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/image_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)
dims = np.array([2,.5,.5])
# instatiate image 3d object
trying = Image3D(lbt_data, dims)
normalised_image = trying.array


#################################################################################
# manually define


parameters = [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9]
trans = []
for para in parameters: 
    trans.append(AffineTransform(para).tM)

images = []
for transformation in trans: 
    im = Image3D(lbt_data,dims).warp3d(transformation)
    images.append(im)
#%%
#slist=[3,9,17,22,25]

#pil_silce(images,slist,"Manual_trans_")
#plot_slices(images,"Manual_trans_/Manual Transformation",slist)

#################################################################################
# generate 10 different randomaly warped images and plot
# 5 image slices for each transformed image at different 
# z depths
#%%

rand_t = []
rand_im = []

for ran in range(10): 
    r_t =AffineTransform(None).tM 
    r_im = Image3D(lbt_data,dims).warp3d(r_t)
    rand_t.append(r_t)
    rand_im.append(r_im)

#pil_silce(rand_im,slist,"Random_trans_")
#plot_slices(rand_im,"Random_trans_/Random transforms",slist=[3,9,17,22,25])

#################################################################################

#%%
# change the strength parameter in random_transform_generator
# generate images with 5 different values for the strength 
# parameter. visualise the randomaly transformed images
r_scal_t = []
r_scal = []
for p in [0.3,0.7,0.5,1,0.8]:
    r_t =AffineTransform(None, st = p).tM 
    r_im = Image3D(lbt_data,np.array([1,1,1])).warp3d(r_t)
    r_scal_t.append(r_t)
    r_scal.append(r_im)

#pil_silce(r_scal,slist,"Rand_sal_trans_")
#plot_slices(r_scal,"Rand_sal_trans_/Strength Parameter",slist=[3,9,17,22,25])

#################################################################################


# %%
