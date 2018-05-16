## Estimating a dense depth map from a sparse depth map using a multi layer perceptron and object segmentation.

Authors:  
    Patrick Henriksen (github: pat676)  
    Vemund Schøyen (github: Vemundss)
    
This project is done as a schoolproject for the subject UNIK4690- Machine Vision

We developed an algorithm for computing a dense depth map using a stereo-geometry sparse depth map as training data for an MLP. The MLP uses pixel coordinates and segmentation as input.

We focused on writing understandable, modular code to ease experimentation with different parameters and methodes. We have not focused on speed while working on this project. 

All modules have a if __name__=='__main__' block at the bottom showing an example 
usage of the code.

### The modules:

#### Matcher.py:
Calculates features and matches these features in stereo images

#### StereoDepth.py
Uses Matcher.py to calculate a sparse detpth

#### PixelSelectGUI.py:
Used to manually select markers for the watershed algorithm in Watershed.py

#### Watershed.py:
Segments a image

#### MLP.py
A tensorflow implementation of a multi layer perceptron

#### MLPDataProsessor.py:
Util functions used to format data before and after use in the MLP

#### MLPDataInterpereter.py:
Util functions used for statistics and data visualization

#### MiddleburyUtil.py:
Util functions for running the middlebury stereo dataset

#### AR & ARGUI
Shows an example usage of the dense depth map

### Citations:

1. D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang, and P. Westling. High-resolution stereo datasets with subpixel-accurate ground truth. In German Conference on Pattern Recognition (GCPR 2014), Münster, Germany, September 2014.

2. Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun.
Vission meets robotics: The KITTI Dataset. International Journal of Robotics Research. 2013

3. David G. Lowe, Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 2004.