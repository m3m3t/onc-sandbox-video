# Dehaze 

ONC has videos and camera located above and below water.  Often, when processing image below water there are suboptimal viewing conditions due to poor lighting conditions, sediment, mechanical malfunctions, etc.  This makes it difficult for ONC scientists to analyze these images for evaluating biodiversity, etc.  

Dehazing is the process of remove haze from an image so objects that are difficult to see are illuminated. 

## Running Locally

1. Update the `params.json` file with your Oceans 2.0 API token.
2. Create a directory `/app/data` and give it write permissions
   * Linux/OSX: 
       ```
       sudo mkdir -p /app/data
       sudo chown <user> /app/data
       ```
   * Windows: Update the `params.json` outPath to a local folder.
3. From the `dehaze` folder
   ```
   python3 dehaze_script.py
   ```

This will downloaded a tarball of 3D camera images to `/app/data` (or whatever you've set your **outPath**), preprocess them using the dehaze algorithm, and update the tarball.

## Dehaze Algorithm 

The algorithm for dehaze is in the main script `dehaze_script.py`.  The tuneable parameters for a) connecting to the API, b) search the API, and c) dehaze algorithm are contained in `params.json`.  

__Params__

- preprocess : Enable preprocessing (Default=True)
- wsize      : Size of kernel/neighbourhood to consider when calculating the dark channel (**Default=19**)
- ratio      : How many "bright" pixels in the dark channel (**Default=0.001**)
- omega      : A percentage of the dark channel for the hazy image normalize by the atmospheric light (**Default=0.98**)
- refine     : Refine the transmission map using a guided filter (**Default=False**)
- t_0        : Since the dark channel tends toward 0, this prameter prevents the image from being too dark (**Default=0.1**)

__Workflow__
1.  The images are preprocessed by taking the zero-mean of the RGB channels and clipping between `[-1,1]` and normalized.
2.  The dark channel (how dark the image is) is calculated by using a minimum filter.
3.  The atmospheric light is calculated by finding the max RGB where the dark channel intensity is at it's brightest
4.  The transmission (how the light is distributed) is calculated by applying omega to the normalized dark channel.
5.  (optional) The transmission map can be refined using an edge-preserved guided filter.  This is the only use of OpenCV in the algorithm, so if you have not installed opencv successfully you can run with `refine=0`.
6.  A non-hazy image is reconstructed using the transmission and atmospheric light. 

**Note:** The chromatic values are often adjusted due to this process so the original colors are often not preserved exactly.
