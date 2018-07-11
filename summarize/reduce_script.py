from onc.onc import ONC

import imageio
import cv2 as cv
import numpy as np
import json 
import shutil

from pch import PCH

def runVideo(video_fn, video_out_fn, opts, debug=False):
    video = imageio.get_reader(video_fn,  'ffmpeg')

    frame_size = (608,800) 
    fps = video.get_meta_data()['fps']
    nframes = video.get_meta_data()['nframes']

    video_out = imageio.get_writer(video_out_fn, fps=fps)

    print(video._meta)
    
    gray_curr = None
    gray_prev = None 

    pch = PCH(**opts["pch"])
    pch.initialize(frame_size, fps)

    sampledFrames = np.arange(0,nframes, opts['samplingRate'])
        
    frame_color = video.get_data(0)
    gray_prev = cv.resize(frame_color, (frame_size[1],frame_size[0])) 
    gray_prev = cv.cvtColor(gray_prev, cv.COLOR_RGB2GRAY) 
    
    for frame_num in sampledFrames:
        try:
            frame_color = video.get_data(frame_num) 
        except:
            print("Can't access frame #", frame_num)
            continue

        gray_curr = cv.resize(frame_color, (frame_size[1],frame_size[0])) 
        gray_curr = cv.cvtColor(gray_curr, cv.COLOR_RGB2GRAY) 

        result = pch.update_model(gray_prev, gray_curr)
        result[result < 255] = 0
        new_motion = np.count_nonzero(result)

        gray_prev = gray_curr

        if new_motion > opts["sampleThreshold"]: 
            from_frame = frame_num - opts["samplingRate"]
            for num in np.arange(from_frame, frame_num):
                video_out.append_data(video.get_data(num))
            video_out.append_data(frame_color)
        
        if frame_num % int(fps)*5 == 0:
            print("- {} seconds done.".format(frame_num / fps))
        
        if debug:
            cv.imshow("result", result)
            cv.imshow("frame", gray_curr)
            cv.waitKey(1)

    video_out.close()
    video.close()
   

with open("params.json", "r") as f:
    params = json.load(f)

    onc_    = params["onc"]
    search_ = params["search"]
    opts    = params["summarize"]
    
    from onc.onc import ONC
    onc = ONC(onc_["token"], 
              onc_["production"],
              onc_["showInfo"], 
              onc_["outPath"])

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

    orders = onc.orderDataProduct(
                query, 
                onc_["maxRetries"], 
                onc_["downloadResultsOnly"], 
                onc_["includeMetadataFile"])

    toDownload = [ (order['url'], order['file']) for order in orders['downloadResults']]
    for url,video_fn in toDownload:
        result = onc.downloadFile(url)
        print("{} downloaded: {}".format(video_fn, result['downloaded']))
        video_out_fn = video_fn.replace("." + search_["extension"], "_summ." + search_["extension"])
        runVideo(video_fn, video_out_fn, opts)
        if not opts["keepOriginal"]:
            shutil.move(video_out_fn, video_fn)
            print("Removing original file ", video_fn)
        print("{} finished.".format(video_fn))
        
