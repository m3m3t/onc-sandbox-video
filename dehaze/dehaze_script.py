#!/usr/env/python3
"""

This script is demonstration of the ONC Oceans 2.0 sandbox.  The program will
access image files from a 3D Camera Array and preprocess them to remove haze
and normalize lighting conditions.
"""


import utils

#Trick to install packages using pip that do not come pre-installed
#on the image. This is shown as an example, as imageio is available
utils.install_and_import("imageio", "imageio")
from utils import imageio

from onc.onc import ONC
import shutil
import os
import numpy as np
from scipy.stats import zscore
from scipy.ndimage.filters import minimum_filter as min_filter
import json

"""
This function calculates the atmospheric light of the image
"""
def atmosphere(img, img_dark, ratio):
    ind = img_dark >= img_dark.max()*(1-ratio) 
    atm = np.max(img[ind], axis=0)[np.newaxis,:] 
    return atm  

"""
This function calculates how the light changes throughout
the image
"""
def transmission(img, at, omega, wsize):
    #t(x) = 1-omega min min(I^C(x) / A^c) 
    t = 1.0 - omega * dark_channel(img / at , wsize )
    return t[:,:,np.newaxis]

"""
This calculates how dark the image is
"""
def dark_channel(img,wsize):
    #J_dark(x) = min_wsize ( min J^c(y) )
    j_min  = np.min(img, axis=2, keepdims=False)
    return min_filter(j_min, size=wsize)

"""
Using the matrixes in the above functions, we reconstruct
the image without the underwater haze
"""
def reconstruct( img, at, t, t_0):
    #J = I(x) - A / max(t, t_min) + A
    J = np.abs(img - at) / np.maximum(t_0, t) + at 
    return (1 - (J - J.min()) / (J.max() - J.min())) * 255

"""
Preprocess the image by performing zero-mean and clip
the values to range [-1,1] and normalize [0,1]
"""
def preprocess(src):
    img = src.copy().astype(np.float32)
    img = zscore(img)
    img = np.clip(img, -1, 1)
    return (img + 1.) / 2.

#Perform dehazing of image
def dehaze(src, opts):
    img = preprocess(src) if opts["preprocess"] else src.copy().astype(np.float32) 

    img_dark = dark_channel(img, opts["wsize"])
    at = atmosphere(img, img_dark, opts["ratio"])
    t = transmission(img, at, opts["omega"], opts["wsize"])
    if opts["refine"]:
        import cv2 as cv
        from cv2.ximgproc import guidedFilter
        t = guidedFilter(cv.cvtColor(src, cv.COLOR_BGR2GRAY), t, 5, 0.85)[:,:,np.newaxis]
    return reconstruct(img, at, t, opts["t_0"])

#Load user defined options from a json file 
print("INFO: Loading user settings from json file")
with open("params.json", "r") as f:
    params = json.load(f)

    onc_    = params["onc"]
    search_ = params["search"]
    opts    = params["dehaze"]

    print("INFO: Connecting to ONC Oceans 2.0")
    onc = ONC(onc_["token"], 
              onc_["production"],
              onc_["showInfo"], 
              onc_["outPath"])

    #Replace this value(s) with the desired camera or 
    #video feeds you want to examine.
    print("INFO: Performing data queries")
    dps = onc.getDataProducts(filters={
            'deviceCode' : search_["deviceCode"]})[0]

    query = {
        'dateFrom'          : search_["dateFrom"], 
        'dateTo'            : search_["dateTo"], 
        'dataProductCode'   : dps['dataProductCode'],
        'extension'         : dps['extension'],
        'deviceCode'        : search_["deviceCode"],
    }
    
    print("INFO: Retrieve the data product info and download.")
    orders = onc.orderDataProduct(
                query, 
                onc_["maxRetries"], 
                onc_["downloadResultsOnly"], 
                onc_["includeMetadataFile"])
    
    downloaded = [ order['file'] for order in orders['downloadResults'] if order['downloaded'] ] 

    print("INFO: Downloaded: ", downloaded)
    outPath = onc_["outPath"]
    for tar_fn in downloaded:
        #The camera array images come in a tarfile (.tar) that we need to extract
        images = utils.untar(tar_fn, outPath, "bmp")
        

        print("{}: Processing tarfile images".format(tar_fn))
        processed = [ dehaze(imageio.imread(os.path.join(outPath, img)), opts) for img in images ]
       
        #If we want to keep the original files than we need to save them somewhere else
        new_tar_fn = tar_fn.replace(".tar", "-dehazed.tar")
        source_dir = os.path.dirname(images[0])
        if opts["keepOriginal"]:
            source_dir = source_dir + "-dehazed"
            os.mkdir(os.path.join(outPath,source_dir))
        
        #Save the images
        for i, img in enumerate(processed):
            basename = os.path.basename(images[i])
            print("{}: Saving {} to {} ...".format(tar_fn, basename, source_dir))
            imageio.imwrite(os.path.join(outPath,source_dir,basename), np.uint8(img))

        #Create the tar file and remove the directories we created
        utils.tar(new_tar_fn, outPath, source_dir)
        
        print("{}: Removing source directory".format(source_dir))
        shutil.rmtree(os.path.join(outPath, os.path.dirname(images[0]))) 
        shutil.rmtree(os.path.join(outPath, source_dir)) 
        
        print("{}:  Finished.".format(tar_fn))

print("INFO: Script finished")

