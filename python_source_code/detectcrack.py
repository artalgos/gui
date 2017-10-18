import numpy as np
from scipy import signal

def fgaussian(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def SOAGK(sigma, theta, rho, size, step):
    realsize = [len(np.arange(start=1, step=step, stop=size[1]+step))+1, len(np.arange(start=1, step=step, stop=size[0]+step))+1]
    kernel = np.zeros(realsize)
    Mtheta = np.array([np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)]).reshape([2,2])
    Mtheta2 = np.array([np.cos(-theta), np.sin(-theta), -np.sin(-theta), np.cos(-theta)]).reshape([2,2])

    for j in range(realsize[1]):
        for i in range(realsize[0]):
            x = (i+1-(realsize[0]+1)/2)*step
            y = (j+1-(realsize[1]+1)/2)*step
            phi = np.array([x, y]).dot(Mtheta2).dot(np.array([rho**2, 0, 0, rho**(-2)]).reshape([2, 2])).dot(Mtheta).dot(np.array([x, y]).reshape([2, 1]))
            temp = (np.cos(theta)*x+np.sin(theta)*y)**2/(rho**(-2)*sigma**2)
            kernel[j,i] = rho**2/sigma**2*(temp-1)/(2*np.pi*sigma**2)* np.exp(-phi/(2*sigma**2));
    kernel = kernel/np.abs(kernel).sum()
    return kernel

def validateFilter(filtered, S, angle, distance, option):
    f = 1
    v = distance
    validated = np.zeros(filtered.shape)
    
    if option==1:
        for j in np.arange(start = v+1, stop = filtered.shape[0]-v):
            for i in np.arange(start = v+1, stop = filtered.shape[1]-v):
                n = [np.cos(angle[j-1, i-1]), np.sin(angle[j-1, i-1])]
                if S[int(np.round(j+v*n[1]))-1, int(np.round(i+v*n[0]))-1] > f*S[j-1,i-1] and \
                S[int(np.round(j-v*n[1]))-1, int(np.round(i-v*n[0]))-1] > f*S[j-1,i-1]:
                    validated[j-1, i-1] = filtered[j-1, i-1]
    else:
        for j in np.arange(start=v+1, stop=filtered.shape[1]-v):
            for i in np.arange(start = v+1, stop = filtered.shape[0]-v):
                n = [np.cos(angle[j-1,i-1]), np.sin(angle[j-1,i-1])]
                if S[int(np.round(j+v*n[1]))-1, int(np.round(i+v*n[0]))-1] < f*S[j-1,i-1] and \
                S[int(np.round(j-v*n[1]))-1, int(np.round(i-v*n[0])-1)] < f*S[j-1,i-1]:
                    validated[j-1, i-1] = filtered[j-1, i-1]
    return(validated)

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def generate_ridges(image, angles, S, A, crackType):
    image = np.array(image)
    img = image[:,:,0]
    img = im2double(img)
    h = fgaussian((4,4), 0.3)
    img = signal.correlate2d(img, h, mode='same')
    if crackType: # 1 = vertical only, 2 = horizontal only
        n = 1
        D = [(crackType - 1)*np.pi/2]
    else: # 0 = multi-directional
        n = angles
        D = np.arange(start=0, stop=np.pi, step=np.pi/n)
    index = 0
    
    orientation_table = np.zeros(n)
    im_filtered = np.zeros((img.shape[0], img.shape[1], n))
    
    for d in range(len(D)):
        kernel = SOAGK(S, D[d], A, [20, 20], 1)
        im_filtered[:,:,index] = signal.convolve2d(img, kernel, mode="same", boundary="symm")
        orientation_table[index] = D[d]
        index = index + 1

    ridge_intensity = np.amax(im_filtered, axis=2)
    angle_index = im_filtered.argmax(axis=2)
    ridge_orientation = orientation_table[angle_index]
    h = fgaussian((5,5), 0.6)
    im_blurred = signal.correlate2d(img, h, mode="same")
    ridge_validated = validateFilter(ridge_intensity, im_blurred, ridge_orientation, 3, 1)
    #ridge_validated = ridge_validated.clip(min=0)

    #ridge_adjusted = 255 - (ridge_validated * 255 / np.max(ridge_validated))

    return ridge_validated







