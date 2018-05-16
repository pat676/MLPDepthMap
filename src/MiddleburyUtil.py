import cv2
import numpy as np
import tensorflow as tf

import Matcher
import MLP
import MLPDataProsessor
import MLPDataInterpreter

class MiddleburyUtil():

    """
    A helper class for working with the Middlebury ground truth pmf files
    """

    def readPmf(self, filepath):

        """
        Reads the middlebury pmf files and returns a numpy array

        Args:
            Filepath(String): The path of the pmf file
        """

        with open(filepath, "rb") as f:

            #Only works for grayscale
            imgType =  f.readline().decode('utf-8').rstrip()
            assert imgType == "Pf", "pmf file not grayscale"
        
            #Read header
            width, height = f.readline().split()
            width, height = int(width), int(height)
            scaleFactor = float(f.readline().rstrip())

            #Determine endian
            endian = '<' if scaleFactor < 0 else '>'

            data = np.fromfile(f, endian + 'f')
        
        img = np.reshape(data, (height, width))
        img = np.flip(img, 0)
        img[img == np.inf] = 0
        return img

    def dispToDepth(self, dispImage, bx, f, doffs):

        """
        Transforms the middlebury disparity image to depth image

        Args:
            dispImage([[float]]): The disparity image
            bx(float)           : The baseline found in Middlebury calib file
            f(float)            : The focal length as found in the Middlebury
                                  calib file
            doffs(flaot)        : x-difference of principal points, found in 
                                  the Middlebury calib file

        """

        return bx*f/(dispImage + doffs)


    def savePng(self, filename, img):
        cv2.imwrite(filename+".png", img)


    def minSegmentEdges(self, img):

        """
        Sets all depth pixels with disparity 0 to the minimum in a 3x3 neigborhood
        
        Args:
            img([[Double]]): The image
        """

        edgeIndicies = np.argwhere(img == 0)
        newImg = img.copy()
        for pt in edgeIndicies:
            x,y = pt

            #Calculate neigborgood
            imgMaxX = img.shape[0]
            imgMaxY = img.shape[1]

            minX = int(x - 1)
            if minX < 0:
                minX = 0

            maxX = int(x + 2)
            if maxX > imgMaxX:
                maxX = imgMaxX

            minY = int(y - 1)
            if minY < 0:
                minY = 0
            
            maxY = int(y + 2)
            if maxY > imgMaxY:
                maxY = imgMaxY

            newImg[x][y] = (img[minX:maxX, minY:maxY][img[minX:maxX, minY:maxY]!=0]).min()

        return newImg

    def compare(self, calculatedImg, groundTruthImage, bx, f, doffs, verbose=True):

        """
        Calculates the difference image and prints statistics

        Args:
            calculatedImg([[Double]])       :The image that should be compared with
                                             the ground truth image
            groundTruthImgage([[Double]])   :The ground truth image
            verbose(Bool)                   :If true statistics are calcualted and
                                             printed to terminal

        """

        #Invalid pixels in GT image (Set from inf to 0 in readpmf)
        calculatedImg[groundTruthImage == 0] = 0 
   
        totalPixels = np.shape(calculatedImg)[0] * np.shape(calculatedImg)[1]
        totalPixels -= (groundTruthImage == 0).sum()
        
        dispDiff = abs(calculatedImg - groundTruthImage)

        calculatedDepthImage = bx*f/(calculatedImg + doffs)
        GTDepthImage = bx*f/(groundTruthImage + doffs)

        depthDiff = abs(calculatedDepthImage - GTDepthImage)    

        #Print the pixel statistics
        if(verbose):
                    
            print("Mean difference in pixels:")
            print(np.mean(dispDiff))
        
            print("Pixel amount with more diff than 0.5 pixels:")
            print((dispDiff > 0.5).sum()/totalPixels)

            print("Pixel amount with more diff than 1 pixels:")
            print((dispDiff > 1).sum()/totalPixels)

            print("Pixel amount with more diff than 2 pixels:")
            print((dispDiff > 2).sum()/totalPixels)

            print("Pixel amount with more diff than 4 pixels:")
            print((dispDiff > 4).sum()/totalPixels)

            print("Pixel amount with more diff than 8 pixels:")
            print((dispDiff > 8).sum()/totalPixels)
        
        if(verbose):

            print("Mean difference in mm:")
            print(np.mean(depthDiff))

            print("Pixel amount with more diff than 25 mm:")
            print((depthDiff > 25).sum()/totalPixels)

            print("Pixel amount with more diff than 50 mm:")
            print((depthDiff > 50).sum()/totalPixels)

            print("Pixel amount with more diff than 100 mm:")
            print((depthDiff > 100).sum()/totalPixels)

            print("Pixel amount with more diff than 200 mm:")
            print((depthDiff > 200).sum()/totalPixels)

            print("Pixel amount with more diff than 400 mm:")
            print((depthDiff > 400).sum()/totalPixels)
        
        return dispDiff

if __name__ == '__main__':

    #Define image paths
    imgLeftPath = "../Images/middelburyLeft.png"
    imgRightPath = "../Images/middelburyRight.png"
    picklePath = "../Pickle/middelburyLeft.p"
    imgLeft = cv2.imread('../Images/middelburyLeft.png')

    matcherFunc = Matcher.SGBFMatcher(imgLeft.shape, lowThres=100, highThres=300, 
                                      epipolarTol=5, localSize=25, localNum=8)

    # Create the MLP data
    mlpDataProsessor = MLPDataProsessor.MLPDataProsessor(imgLeftPath, imgRightPath,
                                                         picklePath=picklePath,
                                                         f=3971.415, bx=195.187,
                                                         doffs=146.53,
                                                         matcherFunc = matcherFunc,
                                                         shouldUseDisp=True,
                                                         edgeRemoveRadius=4
                                                         )
    mlpDataProsessor.normalizeMLPData()
    inputMLP, targetMLP, inputTestMLP, targetTestMLP = mlpDataProsessor.getMLPData()

    # MLP functions
    activationFuncs = [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid]
    optimizerFuncs = [tf.train.AdamOptimizer(0.1),tf.train.AdamOptimizer(0.01),
                     tf.train.AdamOptimizer(0.001),tf.train.AdamOptimizer(0.0001)]
    lossFunc = tf.losses.absolute_difference
     
    # Train the MLP
    mlp = MLP.MLP(len(inputMLP[0]),1,[15,15],optimizerFuncs, activationFuncs, 
                      lossFunc)
    mlp.train(inputMLP, targetMLP, batchSize=100, earlyStopEpochs=3,iterations=75, 
              maxEpochs=500, kFoldValidationSize=0.20, verbose=True, 
              epochsBetweenPrint=10)


    # Run the MLP and denormalize data
    inputArray = mlpDataProsessor.convertSegmentCoordinatesToArray()
    resultDispArrayNormalized = mlp.forward(inputArray)
    resultDispArray = mlpDataProsessor.denormalizeTarget(resultDispArrayNormalized)
    resultDispImage = mlpDataProsessor.convertResultArrayToImage(resultDispArray)
    
    # Compare to ground truth
    middleburyUtil = MiddleburyUtil()
    resultDispImage = middleburyUtil.minSegmentEdges(resultDispImage)
    trueDispImage = middleburyUtil.readPmf("../Images/MiddleburyGroundTruth.pfm")

    #trueDepthImage = middleburyUtil.dispToDepth(trueDispImage, 195.187, 3989.963, 155.41)
    diffImage = middleburyUtil.compare(resultDispImage, trueDispImage, 195.187, 3989.963, 155.41)
    maxDisp = max(trueDispImage.max(), resultDispImage.max())

    # Visualize results
    mlpDataInterpreter = MLPDataInterpreter.MLPDataInterpreter()
    resultDispColorMap = mlpDataInterpreter.applyLogJetColorMap(resultDispImage, 
                                                                logScale=False,
                                                                maxValue=maxDisp)
    resultDispColorMap = mlpDataInterpreter.createColorBarWithLogDepth(maxDisp, 
                                                                       resultDispColorMap,
                                                                       logScale=False,
                                                                       scaleFactor=1)
    trueDispColorMap = mlpDataInterpreter.applyLogJetColorMap(trueDispImage,
                                                              logScale=False,
                                                              maxValue=maxDisp)
    trueDispColorMap = mlpDataInterpreter.createColorBarWithLogDepth(maxDisp, 
                                                                     trueDispColorMap,
                                                                     logScale=False,
                                                                     scaleFactor=1)
    diffColorMap = mlpDataInterpreter.applyLogJetColorMap(diffImage, logScale=False)
    diffColorMap = mlpDataInterpreter.createColorBarWithLogDepth(diffImage.max(), 
                                                                 diffColorMap,
                                                                 logScale=False,
                                                                 scaleFactor=1)

    # Save vizualized results
    cv2.imwrite("../Results/calculatedDisp.png", resultDispColorMap)
    cv2.imwrite("../Results/trueDisp.png", trueDispColorMap)
    cv2.imwrite("../Results/diff.png", diffColorMap)