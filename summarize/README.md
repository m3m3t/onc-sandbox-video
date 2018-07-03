# Video Summarization

ONC deploys video feeds to multiple locations and has accumulated terabytes of video data.  However, not all this video data is useful as it may have poor visibility or no events of interest.  It is a time-consuming process to look through months and years worth of video data.  This is produces a storage problem as a single day of video can produce 3 GB of video data.  One way to automatically handle this is to use video summarization techniques to preprocess the data and remove or sample frames that might be of interest to ONC scientists.

## Running Locally

1. Update the `params.json` file with your Oceans 2.0 API token.
2. Create a directory `/app/data` and give it write permissions
   * Linux/OSX: 
       ```
       sudo mkdir -p /app/data
       sudo chown <user> /app/data
       ```
   * Windows: Update the `params.json` outPath to a local folder.
3. From the `summarize` folder
   ```
   python3 reduce_script.py
   ```

This will downloaded 23 videos of underwater videos from the [Axis P1347 Video Camera](https://data.oceannetworks.ca/Camera?cameraid=12170) at to `/app/data` (or whatever you've set your **outPath**), reduce the videos down.  If `keep_original` is **1** than the summarized video is tagged with *_summ* (each video is between 4-5 minutes).  Otherwise, the original video is overwritten.

## Running Locally (GPU)

There is a GPU-enabled version of the summarization algorithm.  Update `params.json` and set *use_gpu : 1*.  This requires the **numba** and **cuda-toolkit** to be installed.

**NOTE**:  If you get an error than *libNVVM cannot be found*, it's because the numba can't find your CUDA installation.  Add the following (update paths to your CUDA installation) to your `~/.bashrc`:

```
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/nvidia-384
```

## Video Summarization Algorithm 

The algorithm used was by (Dash & Albu)[https://link.springer.com/chapter/10.1007/978-3-319-70353-4_37].  The algorithm models the background using a Gaussion Mixture Model (GMM).  This produces a lot of noise in underwater videos because of current and particulars in the water.  To offset that, per pixel activation and ajustment functions were added.  When a certain number of pixels are activated, the samples frames are added to the output summarized video.  For most cases, roughly 50% of the video is reduced.

__Params__

- samplingRate : The algorithm with analysis every N frames (Default=16)
- sampleThreshold : Number of significant events required to trigger video write (**Default=10**)
- keepOriginal    : Keep original video and tag summarized video with __summ__, otherwise overwrite (**Default=1**)
- learningRate    : The rate the GMM learns (decrease to register smaller events) (**Default=0.1**)
- use_gpu     : Use the GPU for processing the frames (**Default=0**) **NOTE**: Not tested in the sandbox.

__Workflow__
1.  The frame is selected and reduced in size and grayscaled.
2.  The Gaussion Mixture Model is found using the OpenCV MOG function.
3.  The GMM is adjusted based on accumulation and decay rates to take into account noisy backgrounds and clipped to between `[0-1]`
4.  Per pixel adjustment and activation functions are applied so that pixels that have consistent intensity changes between frames are activated as events.  
5.  High level events are counted and if they exceed the sampleThreshold the frame is written to the summary
6. (optional) Once the video is completely analysed, the original is overwritten if keepOriginal is not set.


   
