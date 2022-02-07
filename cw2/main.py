"""I've commented out all the plotting so that it runs nicely"""

"""
Implementation of a reslicing algorithm that obtains
an image slice in a non-orthogonal plane reasonable
for viewing "positioning" parameter of the re-slicing
algorithm?

-   Why do we need such a reslicing algorithm in 
    clinical practice?
-   What are the reasonable "positioning" parameter
    of the the re-slicing algorithm?
-   Visualise re-sliced example in 2D and 3D with 
    varying positioning parameters values. 
"""

"""
Applying a chosen nonlinear filtering for abdominal MR images.
o Motivation and description of the selected algorithms, for both 2D and 3D versions.
o What are the parameters of the filter?
o Analyse the impact due to varying filter parameter values.
o Discuss the computational performance of the filtering.
"""

"""
Compare two approaches, “3D-filtering before re-slicing” and “2D-filtering after re-slicing”.
o Visualise the results from both approaches.
o Qualitative comparison between the two approaches.
o What metrics can be used for quantifying the difference between the two approaches?
o Quantitative comparison between the two approaches.
o Discuss the potential clinical impact due the observed difference.
"""

"""
How can we utilise the organ segmentation to help 1) the filtering and/or 2) the comparison?
"""
#%%
import numpy as np
import SimpleITK as sitk 
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt

# %%
image_id = sitk.ReadImage("data/image.nii.gz")#, imageIO="NiftiImageIO")
image = sitk.GetArrayFromImage(image_id)
vox_dims = image_id.GetSpacing()

#print(image.shape)
#plt.figure()
#plt.imshow(image[12])

# %%
img_subsampled = image[::,::]

#n=12
#plt.figure()
#plt.imshow((image[n]))
#plt.axis('off');

#plt.figure()
#plt.imshow((img_subsampled[n]))
#plt.axis('off');
# %%
def resample_display(image, euc_trans_2d, tx, ty, theta):
    euc_trans_2d.SetTranslation((tx, ty))
    euc_trans_2d.SetAngle(theta)
    resampled_image = sitk.Resample(image, euc_trans_2d)
    plt.imshow(sitk.GetArrayFromImage(resampled_image))
    plt.axis('off')    
    plt.show()

euc_trans_2d = sitk.Euler2DTransform()
euc_trans_2d.SetCenter(image_id.TransformContinuousIndexToPhysicalPoint(np.array(image_id.GetSize())/2.0))
#resample_display(image_id[:,:,45], euc_trans_2d, tx=20.0, ty=64.0, theta= np.pi/12)

# %%
euc_trans_3d = sitk.Euler3DTransform()
euc_trans_3d.SetCenter(image_id.TransformContinuousIndexToPhysicalPoint(np.array(image_id.GetSize())/2.0))
euc_trans_3d.SetRotation(0,np.pi,0)
euc_trans_3d
euc_trans_3d1 = sitk.Euler3DTransform()
euc_trans_3d1.SetCenter(image_id.TransformContinuousIndexToPhysicalPoint(np.array([192,160,256])/2.0))
euc_trans_3d1.SetRotation(0,np.pi+np.pi/5,0)
euc_trans_3d
euc_trans_3d2 = sitk.Euler3DTransform()
euc_trans_3d2.SetCenter(image_id.TransformContinuousIndexToPhysicalPoint(np.array(image_id.GetSize())/2.0))
euc_trans_3d2.SetRotation(0,np.pi+1,0)
euc_trans_3d3 = sitk.Euler3DTransform()
euc_trans_3d3.SetCenter(image_id.TransformContinuousIndexToPhysicalPoint(np.array(image_id.GetSize())/2.0))
euc_trans_3d3.SetRotation(0,np.pi+np.pi/12,0)


resampled_image  = sitk.Resample(image_id, euc_trans_3d)
resampled_image1 = sitk.Resample(image_id, euc_trans_3d1)
resampled_image2 = sitk.Resample(image_id, euc_trans_3d2)
resampled_image3 = sitk.Resample(image_id, euc_trans_3d3)
m= 140
n= 100
l = 100
"""
plt.figure(figsize=(15,15))
plt.subplot(1,4,1)
plt.imshow(sitk.GetArrayFromImage(resampled_image)[m+20,:,:])
plt.xlabel("Rotation = pi")
plt.xticks([]), plt.yticks([])   
plt.subplot(1,4,3)
plt.imshow(sitk.GetArrayFromImage(resampled_image1)[m+20,:,:])
plt.xlabel("Rotation = pi+pi/5")
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow(sitk.GetArrayFromImage(resampled_image2)[m+20,:,:])
plt.xlabel("Rotation = pi+1")
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow(sitk.GetArrayFromImage(resampled_image3)[m+20,:,:])
plt.xlabel("Rotation = pi+pi/12")
plt.xticks([]), plt.yticks([])
#%%
plt.figure(figsize=(15,15))
plt.subplot(1,4,1)
plt.imshow(sitk.GetArrayFromImage(resampled_image)[:,n,:])
plt.xlabel("Rotation = pi")
plt.xticks([]), plt.yticks([])    
plt.subplot(1,4,2)
plt.imshow(sitk.GetArrayFromImage(resampled_image1)[:,n,:])
plt.xlabel("Rotation = pi+pi/5")
plt.xticks([]), plt.yticks([])    
plt.subplot(1,4,3)
plt.imshow(sitk.GetArrayFromImage(resampled_image)[:,:,l])
plt.xlabel("Rotation = pi")
plt.xticks([]), plt.yticks([])    
plt.subplot(1,4,4)
plt.imshow(sitk.GetArrayFromImage(resampled_image1)[:,:,l])
plt.xlabel("Rotation = pi+pi/5")
plt.xticks([]), plt.yticks([])   
plt.show()
"""
#%%

""" Perona Malik Diffusion 2D """

import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

# Presets
# number of times to carry out finit difference, grain level control, diffusion control
#image_file = image_id
#iterations, delta, kappa = 10,1,15

def filtering_2d(image2d, iterations = 10, delta = 1, kappa = 15):
    """Perona-Malik-algorithm
    Inputs:     image2d : a 2d image
                iterations = number of iterations to run finite difference
                delta = grain parameter
                kappa = diffusion control parameter
    
    adapted from github: https://github.com/fubel/PeronaMalikDiffusion/blob/master/main.py
    """
    u = image2d
    dd = np.sqrt(2)
    
    # 2D finite difference windows
    windows = [
        np.array(   [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64        ),
        np.array(   [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64        ),
        np.array(   [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64        ),
        np.array(   [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64        ),
        np.array(   [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64        ),
        np.array(   [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64        ),
        np.array(   [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64        ),
        np.array(   [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64        ), ]

    for r in range(iterations):
        # approximate gradients
        nabla = [ ndimage.filters.convolve(u, w) for w in windows ]

        # approximate diffusion function
        diff = [ 1./(1 + (n/kappa)**2) for n in nabla]

        # update image
        terms = [diff[i]*nabla[i] for i in range(4)]
        terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
        u = u + delta*(sum(terms))

    return(u)

# %%
kappa = 5
iterations = 10
delta = 1
image = resampled_image1
gim = resampled_image2
im = sitk.GetArrayFromImage(image)
g = sitk.GetArrayFromImage(gim)
im_2d = im[140,:,:]
g_2d = g[140,:,:]
"""
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(g_2d)
plt.xlabel("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(filtering_2d(g_2d, iterations, delta, 5))
plt.xlabel("Kappa = 5")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(filtering_2d(g_2d, iterations, delta, 20))
plt.xlabel("Kappa = 20")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(filtering_2d(g_2d, iterations, delta, 40))
plt.xlabel("Kappa = 40")
plt.xticks([]), plt.yticks([])
plt.show()
"""
#%%

# %%

def filtering_3d(image_3d,iterations=1,kappa=50,gamma=0.1,option=1,step=(1.,1.,1.)):
    
    """
    Adapted from https://github.com/awangenh/fastaniso/blob/master/fastaniso.py
    3D Anisotropic diffusion.
    Usage:
    filtering_im = filtering_3d(image_3d, iterations, kappa, gamma)
    Inputs:
            image_3d    = 3 dimentional image to be filtered
            iterations  = number of iterations
            kappa       = conduction coefficient 20-100 ?
            gamma       = max value of .25 for stability
            step        = tuple, the distance between adjacent pixels in (z,y,x)
            option      = 1 Perona Malik diffusion equation No 1
                        = 2 Perona Malik diffusion equation No 2
            
    Returns:
            filtered image = filtering_im   - diffused stack.
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.
    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    image_3d = image_3d.astype('float32')
    filtering_im = image_3d.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(filtering_im)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(filtering_im)
    gE = gS.copy()
    gD = gS.copy()

    for ii in range(iterations):

        # calculate the diffusion
        deltaD[:-1,: ,:  ] = np.diff(filtering_im,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(filtering_im,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(filtering_im,axis=2)

        # gradients 
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one pixel
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        filtering_im += gamma*(UD+NS+EW)

    return filtering_im
# %%
"""
#perona(im[:,:,110],g[:,:,89,])
d3 = filtering_3d(im) 
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(d3[140])
plt.xticks([]), plt.yticks([])

# %%
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(g_2d)
plt.xlabel("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(filtering_3d(g, iterations, 5) [140,:,:])
plt.xlabel("Kappa = 5")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(filtering_3d(g, iterations, 20)[140,:,:] )
plt.xlabel("Kappa = 20")
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(filtering_3d(g, iterations, 40)[140,:,:] )
plt.xlabel("Kappa = 40")
plt.xticks([]), plt.yticks([])
plt.show()
# %%


gim = resampled_image2
g = sitk.GetArrayFromImage(gim)

rt = filtering_2d(g[140,:,:], iterations, kappa = 7) 

rt3 = filtering_3d(g, iterations, 7)[140,:,:] 

norm = np.hypot(rt3,rt)
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('2d, Kappa = 40')
plt.subplot(1, 4, 2), plt.imshow(rt3)
plt.xticks([]), plt.yticks([])
plt.xlabel('3d, Kappa = 40')
plt.subplot(1, 4, 3), plt.imshow(norm)
plt.xticks([]), plt.yticks([])
plt.xlabel('Norm')
plt.subplot(1, 4, 4), plt.imshow(rt3-rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('Difference')
plt.show()

print(np.mean(rt3-rt))
print(np.max(rt3))
print(np.min(rt3))
print(np.mean(rt3))
print(np.std(rt3))

print(np.max(rt))
print(np.min(rt))
print(np.mean(rt))
print(np.std(rt))


rt = filtering_2d(g[140,:,:], iterations, kappa =15) 

rt3 = filtering_3d(g, iterations, 60)[140,:,:] 

norm = np.hypot(rt3,rt)
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('2d, Kappa = 5')
plt.subplot(1, 4, 2), plt.imshow(rt3)
plt.xticks([]), plt.yticks([])
plt.xlabel('3d, Kappa = 50')
plt.subplot(1, 4, 3), plt.imshow(norm)
plt.xticks([]), plt.yticks([])
plt.xlabel('Norm')
plt.subplot(1, 4, 4), plt.imshow(rt3-rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('Difference')
plt.show()
# %%
print(np.mean(rt3-rt))
print(np.max(rt3))
print(np.min(rt3))
print(np.mean(rt3))
print(np.std(rt3))

print(np.max(rt))
print(np.min(rt))
print(np.mean(rt))
print(np.std(rt))
# %%

gim = resampled_image2
g = sitk.GetArrayFromImage(gim)

rt = filtering_2d(g[90,:,:], iterations, kappa = 7) 

rt3 = filtering_3d(g, iterations, 7)[90,:,:] 

norm = np.hypot(rt3,rt)
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('2d, Kappa = 40')
plt.subplot(1, 4, 2), plt.imshow(rt3)
plt.xticks([]), plt.yticks([])
plt.xlabel('3d, Kappa = 40')
plt.subplot(1, 4, 3), plt.imshow(norm)
plt.xticks([]), plt.yticks([])
plt.xlabel('Norm')
plt.subplot(1, 4, 4), plt.imshow(rt3-rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('Difference')
plt.show()

print(np.mean(rt3-rt))
print(np.max(rt3))
print(np.min(rt3))
print(np.mean(rt3))
print(np.std(rt))

print(np.max(rt))
print(np.min(rt))
print(np.mean(rt))
print(np.std(rt))


rt = filtering_2d(g[90,:,:], iterations, kappa = 15) 

rt3 = filtering_3d(g, iterations, 60)[90,:,:] 

norm = np.hypot(rt3,rt)
plt.figure(figsize=(15,15))
plt.subplot(1, 4, 1), plt.imshow(rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('2d, Kappa = 5')
plt.subplot(1, 4, 2), plt.imshow(rt3)
plt.xticks([]), plt.yticks([])
plt.xlabel('3d, Kappa = 50')
plt.subplot(1, 4, 3), plt.imshow(norm)
plt.xticks([]), plt.yticks([])
plt.xlabel('Norm')
plt.subplot(1, 4, 4), plt.imshow(rt3-rt)
plt.xticks([]), plt.yticks([])
plt.xlabel('Difference')
plt.show()
# %%
print(np.mean(rt3-rt))
print(np.max(rt3))
print(np.min(rt3))
print(np.mean(rt3))
print(np.std(rt))

print(np.max(rt))
print(np.min(rt))
print(np.mean(rt))
print(np.std(rt))

# %%
"""