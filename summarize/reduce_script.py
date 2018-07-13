
#!/usr/env/python3
"""

This script is demonstration of the ONC Oceans 2.0 sandbox.  The program will access video files from an underwater video camera and reduce the videos duration by roughly 50 percent to save on space and watch time. The videos are reduced based on "novelty" which is modelled using background subtraction and historical repetition.
"""

from onc.onc import ONC

import imageio
import cv2 as cv
import numpy as np
import json 
import shutil
import sys

from pch import PCH

"""
Run through the video and only save frames that exhibit
a certain amount of novelty that modelled using the PCH
algorithm.
"""
def runVideo(video_fn, video_out_fn, opts, debug=False):
    video = imageio.get_reader(video_fn,  'ffmpeg')

    #To speed up processing, we resize the video to smaller
    #dimensions
    frame_size = (608,800) 
    fps = video.get_meta_data()['fps']
    nframes = video.get_meta_data()['nframes']

    #We'll use the same frames per second as the input video
    video_out = imageio.get_writer(video_out_fn, fps=fps)

    print(video._meta)
    
    gray_curr = None
    gray_prev = None 

    #Create our novelty model
    pch = PCH(**opts["pch"])
    pch.initialize(frame_size, fps)

    #Another time saving measure, we're only going to sample
    #some of the videos for analysis
    sampledFrames = np.arange(0,nframes, opts['samplingRate'])
        
    frame_color = video.get_data(0)
    gray_prev = cv.resize(frame_color, (frame_size[1],frame_size[0])) 
    gray_prev = cv.cvtColor(gray_prev, cv.COLOR_RGB2GRAY) 
    
    for frame_num in sampledFrames:
        try:
            frame_color = video.get_data(frame_num) 
        except:
            print("Can't access frame #", frame_num, file=sys.stderr)
            continue

        gray_curr = cv.resize(frame_color, (frame_size[1],frame_size[0])) 
        gray_curr = cv.cvtColor(gray_curr, cv.COLOR_RGB2GRAY) 

        #Given the underwater is murky, we'll only consider high novel events (255). 
        result = pch.update_model(gray_prev, gray_curr)
        result[result < 255] = 0
        new_motion = np.count_nonzero(result)

        gray_prev = gray_curr

        #To prevent noise, we'll use a threshold for the number of events. Since we always resize videos, we can use a set value
        if new_motion > opts["sampleThreshold"]: 
            #To prevent video "fast forwarding" we'll go back and write the frames that we skipped during sampling.  This adds a bit of time back but still better than not sampling.
            from_frame = frame_num - opts["samplingRate"]
            for num in np.arange(from_frame, frame_num):
                video_out.append_data(video.get_data(num))
            video_out.append_data(frame_color)
        
        if frame_num % int(fps)*5 == 0:
            print("- {} seconds done.".format(frame_num / fps))
       
        #Show video during debugging locally
        if debug:
            cv.imshow("result", result)
            cv.imshow("frame", gray_curr)
            cv.waitKey(1)

    video_out.close()
    video.close()
 
    #To prevent any memory leaks
    del pch

#Load user defined options from a json file 
print("INFO: Loading user settings from json file")
with open("params.json", "r") as f:
    params = json.load(f)

    onc_    = params["onc"]
    search_ = params["search"]
    opts    = params["summarize"]
    
    print("INFO: Connecting to ONC Oceans 2.0")
    onc = ONC(onc_["token"], 
              onc_["production"],
              onc_["showInfo"], 
              onc_["outPath"])

    #Replace this value(s) with the desired video feeds 
    #you want to examine.
    print("INFO: Performing data queries")
    dps = onc.getDataProducts(filters={
                'deviceCode'   : search_["deviceCode"], 
                'locationCode' : search_["locationCode"], 
                'extension'    : search_["extension"]})[0]
    query = {
        'dateFrom'          : search_["dateFrom"], 
        'dateTo'            : search_["dateTo"], 
        'dataProductCode'   : dps['dataProductCode'],
        'extension'         : dps['extension'],
        'deviceCode'        : search_["deviceCode"],
    }

    #We don't want to download at once in case we have a large amount of video files, so we just retrieve the url files
    print("INFO: Retrieve the data product info")
    orders = onc.orderDataProduct(
                query, 
                onc_["maxRetries"], 
                onc_["downloadResultsOnly"], 
                onc_["includeMetadataFile"])

    toDownload = [ (order['url'], order['file']) for order in orders['downloadResults']]
    for url,video_fn in toDownload:
        result = onc.downloadFile(url)
        print("{} downloaded: {}".format(video_fn, result['downloaded']))
       
        print("{}: Reducing video to important events".format(video_fn))
        video_out_fn = video_fn.replace("." + search_["extension"], "_summ." + search_["extension"])
        runVideo(video_fn, video_out_fn, opts)
       
        #Most of the video is not important, so we don't want to keep the original video to save space.
        if not opts["keepOriginal"]:
            shutil.move(video_out_fn, video_fn)
            print("{}: Removing original file ".format(video_fn))
        print("{}:  Finished.".format(video_fn))

print("INFO: Script finished")
