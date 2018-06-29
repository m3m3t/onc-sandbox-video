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

The algorithm used was [TODO].
   
