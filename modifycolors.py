from os import read
from PIL import Image, ImageSequence
from skimage import color, data, restoration 
from scipy.signal import convolve2d
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time
import io
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def runfilters(im):
    pixdata = im.load()                  
    for y in range(im.size[1]):
        # anything purp make red
        for x in range(im.size[0]):
            if (pixdata[x,y][0] > 120 and pixdata[x,y][1] > 50 and pixdata[x,y][2] > 50) and (pixdata[x,y][0] < 150 and pixdata[x,y][1] < 255 and pixdata[x,y][2] < 255):
                output = (255,0,0,255)
                im.putpixel((x,y), output)

    #anything black make white
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            if False: # (pixdata[x,y][0] < 65 and pixdata[x,y][1] < 65 and pixdata[x,y][2] <65):
                output = (255,255,255,255)
                im.putpixel((x,y), output)

    # anything tooo close to each other make grey
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            if False: #  (pixdata[x,y][0] - pixdata[x,y][1] < 8):
                output = (100,100,100,255)
                im.putpixel((x,y), output)
    
    return im

# simlar as runfilters but on an numpy array
def runfilters2(im):
    # highligt all purple 
    red, green, blue, alpha = im[:,:,0], im[:,:,1], im[:,:,2], im[:,:,3]
    mask = (red > 80) & (green < 80) & (blue > 80) 
    im[:,:,:4][mask] = [255,0,0,255]
    return im
    #anything black make white
    # anything tooo close to each other make grey

def sharpen(im):
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    cp_img = imgs.copy()
    sharpened_array = np.array(cp_img)
    sharpened_array[:,:,0] = convolve2d(sharpened_array[:-2,:-2,0], kernel)
    sharpened_array[:,:,1] = convolve2d(sharpened_array[:-2,:-2,1], kernel)
    sharpened_array[:,:,2] = convolve2d(sharpened_array[:-2,:-2,2], kernel)
    
    # back
    return Image.fromarray(sharpened_array)

# yoink https://note.nkmk.me/en/python-pillow-concat-images/
def get_concat_h(im1, im2):
    dst = Image.new('RGBA', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def normalize_image(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    return img

# run a highpass filter on each color channel 
def highpassfilter(img):
    cp_img = imgs.copy()
    highpass_array = np.array(cp_img)
    to_sharpen = np.array(cp_img)
    
    #grey = cv2.cvtColor(highpass_array, cv2.COLOR_BGR2GRAY) 
    #
    #blur = cv2.GaussianBlur(grey, (21,21),0)
    #high_pass = cv2.subtract(grey, blur)
    #high_pass = cv2.normalize(high_pass, None, 0,255, cv2.NORM_MINMAX)
    #shrpened_img = cv2.addWeighted(to_sharpen, 1.5, cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR), -0.5, 0)
    #return Image.fromarray(shrpened_img)
    blurred   = cv2.GaussianBlur(highpass_array[:,:,0], (21,21),0)
    blurgreen = cv2.GaussianBlur(highpass_array[:,:,1], (21,21),0)
    blurblue  = cv2.GaussianBlur(highpass_array[:,:,2], (21,21),0)
    highpass_array[:,:,0] = cv2.subtract(highpass_array[:,:,0], blurred)
    highpass_array[:,:,1] = cv2.subtract(highpass_array[:,:,1], blurgreen)
    highpass_array[:,:,2] = cv2.subtract(highpass_array[:,:,2], blurblue)
    #normalize 
    highpass_array[:,:,0] = cv2.normalize(highpass_array[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
    highpass_array[:,:,1] = cv2.normalize(highpass_array[:,:,1], None, 0, 255, cv2.NORM_MINMAX)
    highpass_array[:,:,2] = cv2.normalize(highpass_array[:,:,2], None, 0, 255, cv2.NORM_MINMAX)
    
    # combine with og image
    sharpened = cv2.addWeighted(to_sharpen, 1.5,highpass_array, -0.5, 0)
    return Image.fromarray(sharpened)

def laplacefilter(img):
    cp_img = imgs.copy()
    highpass_array = np.array(cp_img)
    to_sharpen = np.array(cp_img)
    
    #grey = cv2.cvtColor(highpass_array, cv2.COLOR_BGR2GRAY) 
    #
    #blur = cv2.GaussianBlur(grey, (21,21),0)
    #high_pass = cv2.subtract(grey, blur)
    #high_pass = cv2.normalize(high_pass, None, 0,255, cv2.NORM_MINMAX)
    #shrpened_img = cv2.addWeighted(to_sharpen, 1.5, cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR), -0.5, 0)
    #return Image.fromarray(shrpened_img)
    blurred   = cv2.Laplacian(highpass_array[:,:,0], cv2.CV_64F)
    blurgreen = cv2.Laplacian(highpass_array[:,:,1], cv2.CV_64F)
    blurblue  = cv2.Laplacian(highpass_array[:,:,2], cv2.CV_64F)
    #normalize 
    highpass_array[:,:,0] = cv2.normalize(highpass_array[:,:,0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    highpass_array[:,:,1] = cv2.normalize(highpass_array[:,:,1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    highpass_array[:,:,2] = cv2.normalize(highpass_array[:,:,2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # combine with og image
    sharpened = cv2.addWeighted(to_sharpen, 1.0, highpass_array, .5, 0)
    return Image.fromarray(sharpened)

def filter_then_morph(img):
    # cv2 morph open then close
    arr = np.array(img.copy())
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    
    # Define the range for the color purple in HSV
    lower_purple = np.array([120, 70, 70])
    upper_purple = np.array([150, 255, 255])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = np.array(img.copy())
    cv2.drawContours(output_img, contours, -1,(0, 255, 0), 2)
    #cv2.drawMarker(output_image )
    return Image.fromarray(mask)
    

# 1  google "facebook" 
# 2 look at example code https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # jank way to remove the background mask
    sorted_anns = sorted_anns[1:]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        #color_mask = np.concatenate([(255,0,0), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# 3 see what happens
def segment_anything(img):
    print("runme")
    image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())
    fig = plt.figure(figsize=(25,25))
    plt.axis('off')
    plt.imshow(img)
    anns = show_anns(masks)
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    return Image.open(buf)
    
    


    


    

# getting the runtime for shtis
starttime = time.time()


# convolute with a sharpening kernel

# open each slide 
#images = Image.open("C:\\Users\\henry\\source\\repos\\modifycolors\\modifycolors\\WM4235_1kPa_Motility_RGB.tif")
#images = Image.open("C:\\Users\\henry\\source\\repos\\modifycolors\\modifycolors\\WM4235_1kPa_Invasion001.tif")
images = Image.open("C:\\Users\\henry\\source\\repos\\modifycolors\\modifycolors\\WM4235_25kPa_Invasion002.tif")
# facebook example 
#facebook = Image.open("C:\\Users\\henry\\source\\repos\\modifycolors\\modifycolors\\dog.jpg")




for i, img in enumerate(ImageSequence.Iterator(images)):
    imgs = img.convert('RGBA')
    im2 =segment_anything(img)
    final = get_concat_h(imgs,im2)
    final.show()
    #im2 = runfilters(sharpen(imgs))
    #im2 = filter_then_morph(highpassfilter(imgs.copy()))
    
 




# get run time in seconds
print(time.time() - starttime)
plt.show()





exit()
#TODO? highpass filter
#fig, ax = plt.subplots(nrows=images.n_frames, ncols=2)
#fig.set_dpi(500)

# sharpen each color 
#sharpened_img.show()
#runfilters(sharpened_img, "sharp and filter").show()
#runfilters(imgs, "og image").show()
#@    im1 = runfilters(imgs)
#@    im1.show()
#@    ax[i][1].imshow(im1)
#@    ax[i][1].set_title("filters only")
#@    
#@
#@    
#@    im2 = sharpen(imgs)
#@    ax[i][2].imshow(im2)
#@    ax[i][2].set_title("sharpend 1: convolution")
    
   



#imgs_grey = imgs.convert('L')
#img_array = np.array(imgs_grey)
#for i in range(3):
#    readSlide(imgs.seek(i))
# lets break it out 

# how about deconvolution ? 
# trying the example first
#https://scikit-image.org/docs/stable/auto_examples/filters/plot_deconvolution.html

#psf = np.ones((5,5)) / 25000
#deconvolved = restoration.richardson_lucy(img_array, psf, num_iter=30)

#normalize
#deconvolved = normalize_image(deconvolved)
#seems like not a way forward

            