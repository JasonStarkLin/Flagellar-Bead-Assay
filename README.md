<h1 align="center">
<img src="BeadsAssayLogo.png" width="300">
</h1><br>

Here collect my programs for analyzing the beads assay in the bacterial flagellar motor experiment.

## Features:
-  Automatic recognition of the rotating beads in video files (.seq, .tif).
-  Export ROIs of the recognized beads in ImageJ format.
-  Batch process for the same type of files in the assigned folder.
-  Draw the trace of the rotating orbit and the power spectrum of FFT results.
-  Output the results in a CSV file, including speed, FFT peak, and ellipse fitting quality for further data filtration.

## Analyzing steps:
<img src="Structure.jpg" width="300">
1. Set the parameters.
2. Search the files to analyze. (The program can only analyze video as a stack of file.)
3. Loop the files.
   1. Segment the likely rotating beads by standard deviation of the intensity. And label the region of interest. (Output the labeled image, and ROIs)
   2. Export the information of labled beads in each video file.
   3. Loop the beads.
      1. Localize the bead's position in each frame by calling GetBeadsPosition function.
      2. Export the bead's orbit.
4. Export the summary results for all the beads.
