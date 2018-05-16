import numpy as np 
import cv2
import StereoDepth
import Matcher
import Watershed
import tensorflow as tf
import MLP
import MLPDataInterpreter
import time

class MLPDataProsessor():

    """
    A util class for manipulating the image data before use in the MLP

    Args:
        imgLeftPath([[Double]])          : The path to the left image
        imgRightPath(([Double]))         : The path to the right image
        usedSegments([Int])              : The segemnts that should be used,
                                           if empty all segments will be used.
        featureFunc("cv2.xfeatures2d")   : The function used by Matcher to 
                                           to detect features
        descriptorFunc("cv2.xfeatures2d"): The function used by Matcher to
                                           describe the features
        matcherFunc("cv2.BFMatcher")     : The function used to by Matcher
                                           to match descriptors
        numFeatures(Int)                 : The maximum amount of features that 
                                           should be calculated
        f(float)                         : The camera focal length
        bx(float)                        : The baseline of the cameras                 
        doffs(float)                     : C1x - C0x (Usually 0)
        shouldUseDisp(Bool)              : If true disparities are used instead of
                                           depths
        clipPercentile(float)            : The data is sorted and (1-clipPercentil)
                                           of the closest and furthest data is 
                                           removed
        picklePath(String)               : Load/save segmented image from/to this file
        shouldLoadPickle(Bool)           : If true the segmented image is loaded from
                                           picklePath, else the segmented image
                                           is saved to pickle path
        epipolarTol(float)               : The tolerance used for epipolar test
        ratioThres(float)                : The threshold for the ratio test 
    
    Class variables:
        inputMLP([[Double]])        : A vector with pixel coordinates, and segments
                                      in a "one-hot" encoding, with correspoding 
                                      depths stored in targetMLP
        targetMLP([[Double]])       : The depths corresponding to the innputMLP
        inputTestMLP(([Double]))    : The test set, initialised if 
                                      self.splitValid() is called
        targetTestMLP(([Double]))   : The test sets targets, initialised if 
                                      self.splitValid() is called
        segments([[Int]])           : The segment image matrix
        edgeRemoveRadiu(int)        : Points in the sparse set within this radius
                                      of a segment edge are removed
    """

    def __init__(self, imgLeftPath, imgRightPath, usedSegments=[], featureFunc=None, 
                 descriptorFunc=None, matcherFunc=None, f=None, 
                 bx=None, doffs=0, shouldUseDisp=False, clipPercentile=1, 
                 picklePath=None, shouldLoadPickle = True, epipolarTol = 0.7,
                 ratioThres=0.8, edgeRemoveRadius=1):

        self.imgLeftPath = imgLeftPath
        self.imgRightPath = imgRightPath
        self.imgLeft = cv2.imread(imgLeftPath)
        self.imgRight = cv2.imread(imgRightPath)
        self.edgeRemoveRadius = edgeRemoveRadius

        self.inputMLP, self.targetMLP = [], []
        self.inputTestMLP, self.targetTestMLP = [],[]

        self._createMatcher(featureFunc, descriptorFunc, matcherFunc, epipolarTol, 
                            ratioThres)
        self._calculateSegments(picklePath, shouldLoadPickle)

        #If usedSegments is not given, use all segments
        if(len(usedSegments) == 0):
            usedSegments = list(range(self.numSegments+1))

        self._calculateStereoDepth(f, bx, doffs)
        self._createMLPData(usedSegments, shouldUseDisp)
        if clipPercentile < 1 and clipPercentile > 0:
            self._clipMLPData(clipPercentile, usedSegments)

    def _createMatcher(self, featureFunc, descriptorFunc, matcherFunc, 
                       epipolarTol, ratioThres):

        """
        Creates the Matcher object and runs all computations
        
        If the functions arn't given as args, sift.detect, sift.compute and
        BFMatcher.knnMatch are used

        Args:
            Same as described in class description
        """

        #Create standard functions if not spesified by class var

        self.matcher = Matcher.Matcher(self.imgLeft , self.imgRight, 
                                       featureFunc, descriptorFunc, matcherFunc, 
                                       epipolarTol=epipolarTol,
                                       ratioThres=ratioThres)
        self.matcher.computeAll()
        self.goodMatches = self.matcher.goodMatches
        self.keypointsLeft = self.matcher.keypointsLeft
        self.keypointsRight = self.matcher.keypointsRight

    def _calculateSegments(self, picklePath, shouldLoadPickle):

        """
        Loads the segment image if shouldLoadPickle and picklePath is given,
        else the GUI for segment creation is started

        Args:
            picklePath(String)     : The path of the pickle file to load/save
            shouldLoadPickle(Bool) : If true, pickle file is loded
        """
        
        self.watershed = Watershed.Watershed(self.imgLeftPath)

        if shouldLoadPickle:
            self.segments = self.watershed.loadSegmentedImg(picklePath=picklePath)
        elif picklePath != None:
            self.segments = self.watershed.segmentAndSaveImg(picklePath=picklePath)
        else:
            self.segments = self.watershed.segment()
        self.numSegments = self.segments.max()

    def _calculateStereoDepth(self, f, bx, doffs):

        """
        Calculates the sparse depth mage using StereoDepth class

        Args:
            f (float): The focal length
            bx (float): The baseline
        """
        
        #Use f, bx from the residental Kitty image set if not given by class var 
        if(f == None):
            f = 7.215377e+02
    
        if(bx == None):
            bx = 4.485728e+01 + 3.395242e+02


        self.stereodepth = StereoDepth.StereoDepth(self.goodMatches, self.keypointsLeft, 
                                                   self.keypointsRight, f, bx, doffs)
        self.stereodepth.computeAll()

    def _createMLPData(self, usedSegments, shouldUseDisp):

        """
        Creates the inputMLP and targetMLP arrays

        The first to elements of the input MLP are the coordinates, the
        rest are the "one-hot" encoding for segment number

        Args:
            usedSegments([Int]) : An array with the segments that should be used
                                  for the input/target data
        """

        self.inputMLP = []
        if(shouldUseDisp):
            self.targetMLP = self.stereodepth.disparities
        else: 
            self.targetMLP = self.stereodepth.depths

        numMisMatches = 0 #depth found at edge of segments
        for i in range(len(self.stereodepth.depths)):
            x = int(self.keypointsLeft[self.goodMatches[i][0].queryIdx].pt[1])
            y = int(self.keypointsLeft[self.goodMatches[i][0].queryIdx].pt[0])

            #Remove if (x,y) not in used segment or depth is negative
            if((not self.segments[x][y] in usedSegments) or 
               (self.targetMLP[i-numMisMatches] < 0)):

                del self.targetMLP[i-numMisMatches]
                numMisMatches += 1
                continue

            #Calculate neigborgood
            imgMaxX = self.imgLeft.shape[0]
            imgMaxY = self.imgLeft.shape[1]

            minX = int(x - self.edgeRemoveRadius)
            if minX < 0:
                minX = 0

            maxX = int(x + self.edgeRemoveRadius + 1)
            if maxX > imgMaxX:
                maxX = imgMaxX

            minY = int(y - self.edgeRemoveRadius)
            if minY < 0:
                minY = 0
            
            maxY = int(y + self.edgeRemoveRadius + 1)
            if maxY > imgMaxY:
                maxY = imgMaxY

            #Remove datapoints close to segment border
            if((self.segments[minX:maxX, minY:maxY] == -1).sum() > 0):
                del self.targetMLP[i-numMisMatches]
                numMisMatches += 1
                continue

            self.inputMLP.append(np.zeros(2+self.numSegments))
            self.inputMLP[i-numMisMatches][0] = x
            self.inputMLP[i-numMisMatches][1] = y
            self.inputMLP[i-numMisMatches][self.segments[x][y]+1] = 1   #One-hot encoding of segments

            self.targetMLP[i-numMisMatches] = [self.targetMLP[i-numMisMatches]]
        self._calculateNormalizedConstants()

    def _calculateNormalizedConstants(self):

        """
        Stores the constants used for normalizing the data as class variables
        """

        self.maxXBeforeNormalizing = np.shape(self.segments)[0]
        self.maxYBeforeNormalizing = np.shape(self.segments)[1]
        
        #Adds a small amount to max/min target so MLP output can be higher than 
        #training data
        self.minTargetBeforeNormalizing = np.asarray(self.targetMLP).min()*0.8
        self.maxTargetBeforeNormalizing = np.asarray(self.targetMLP).max()*1.2

    def _clipMLPData(self, percentile, clipSegments):

        """
        Removes the (1-percentile)/2 data with maximum and minimum depth from the
        segments given in clipSegments.

        Args:
            precentil(Double): The percent of the data that should be kept
            clipSegemts(Int) : The segments that are affected
        """

        for seg in clipSegments:

            segmentData = []

            #List the segments index in targetMLP and its depth
            for i in range(0,len(self.targetMLP)):
                if self.inputMLP[i][seg+1] == 1:
                    segmentData.append([i,self.targetMLP[i]])

            segmentData = sorted(segmentData, key=lambda segmentData:segmentData[1][0])
            numToRemove = int((1-percentile)*len(segmentData))
            removeIndexes = []

            #Create a list with indexes to remove
            i = 0
            maxIdx = len(segmentData) - 1
            while(i*2 < numToRemove):
                removeIndexes.append(segmentData[i][0])
                removeIndexes.append(segmentData[maxIdx-i][0])
                i += 1

            #Remove indexes
            for idx in sorted(removeIndexes, reverse=True):
                self.targetMLP.pop(idx)
                self.inputMLP.pop(idx)
            
        self._calculateNormalizedConstants()

    def normalizeMLPData(self):

        """
        Normalizes the coordinates of inputMLP and the depths in targetMLP
        """

        self.inputMLP = np.asarray(self.inputMLP)
        self.inputMLP[:,0] = self.inputMLP[:,0]/self.maxXBeforeNormalizing
        self.inputMLP[:,1] = self.inputMLP[:,1]/self.maxYBeforeNormalizing
        self.inputMLP = list(self.inputMLP)

        targetDiff = self.maxTargetBeforeNormalizing - self.minTargetBeforeNormalizing
        self.targetMLP = (self.targetMLP - self.minTargetBeforeNormalizing)/targetDiff
        self.targetMLP = list(self.targetMLP)
    
    def denormalizeTarget(self, targetNormalized):

        """
        Denormalises target array

        Args:
            targetNormalized([Double]): The normalized target array
        """

        target = targetNormalized.copy()
        targetDiff = self.maxTargetBeforeNormalizing - self.minTargetBeforeNormalizing
        target = (np.asarray(target)*targetDiff) + self.minTargetBeforeNormalizing
        return target 
    
    def splitTest(self, amount):

        """
        Creates a test set. The elements used in the test set 
        are removed from the input/targetMLP sets
        """

        if(amount <= 0 or amount >=1):
            print("ERROR: validtation amount should be a number between 0 and 1")

        testNum = int(len(self.inputMLP)*amount)
        idx = np.random.choice(np.arange(len(self.inputMLP)), testNum, replace=False)

        self.inputTestMLP, self.targetTestMLP = [], []

        c = 0
        for i in idx:
            self.inputTestMLP.append(self.inputMLP.pop(i-c))
            self.targetTestMLP.append(self.targetMLP.pop(i-c))
            c += 1

    def getMLPData(self):

        """
        Returns the MLP Data
        """

        return (self.inputMLP.copy(), self.targetMLP.copy(), 
                self.inputTestMLP.copy(), self.targetTestMLP.copy())

    def convertSegmentCoordinatesToArray(self, usedSegments=[]):
        
        """
        Converts the coordinates of the given segments to an array formated for
        use with the mlp

        Args:
            usedSegments([Int]): The segments that should be formated

        Returns([[Int]]):
            The normalized coordinate array
        """

        if(len(usedSegments) == 0):
            usedSegments = list(range(self.numSegments+1))

        totalCount = 0
        maxX, maxY = np.shape(self.segments)

        #Add [x,y] coordinates belonging to usedSegments
        coordsList = np.nonzero(np.isin(self.segments, usedSegments))
        coords = np.asarray(coordsList, dtype=np.float32).T

        coords[:,0] = coords[:,0]/self.maxXBeforeNormalizing
        coords[:,1] = coords[:,1]/self.maxYBeforeNormalizing

        totalCount = coords.shape[0]

        oneHot = np.zeros((totalCount, self.numSegments))
        oneHot[np.arange(totalCount),self.segments[coordsList]-1] = 1
        segmentArray = np.concatenate((coords, oneHot), axis=1)
        
        return segmentArray

    def convertResultArrayToImage(self, array, usedSegments=[]):

        """
        Converts the depth array resulting from mlp.forward back to a image

        Args:
            usedSegments([Int]) : The segments used creating the array
            array([Double])     : The result array from the mlp containing
                                  the depths
        
        Returns([[Double]]):
            The resulting image
        """

        if(len(usedSegments) == 0):
            usedSegments = list(range(self.numSegments+1))

        maxX, maxY = np.shape(self.segments)
        segmentImg = np.zeros((maxX, maxY))

        coordsList = np.nonzero(np.isin(self.segments, usedSegments))
        segmentImg[coordsList] = array[:,0]
        
        return segmentImg


    def drawSegment(self, segments):

        """
        Draws the selected segments. All pixels not in the given segments
        array are set to 0.

        Args:  
            usedSegments([Int]) : The segments that should be drawn
        
        Returns([[Double]]):
            The image matrix    
        """

        img = cv2.imread(self.imgLeftPath)
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segImg = np.zeros(np.shape(grayImg))

        maxX, maxY = np.shape(grayImg)
        for x in range(0,maxX):
            for y in range(0,maxY):
                if self.segments[x][y] in segments:
                    segImg[x][y] = grayImg[x][y]

        return segImg

    def drawTargets(self, img, scaleFactor=0.001):

        """
        marks the given targets in the image

        Args:
            img([[Double]])     : The image where the targest should be drawn
            scaleFactor(float)  : The depths will be scaled by this factor

        Returns([[Double]]):
            The image with the drawn depths
        """

        depthImg = img.copy()
        denormTarget = self.denormalizeTarget(self.targetMLP)
        for i in range(0,len(self.targetMLP)):
            
            depth = denormTarget[i]*scaleFactor
            text = "{0:.1f}".format(depth[0])
            coord = (int(self.inputMLP[i][1]*self.maxYBeforeNormalizing), 
                     int(self.inputMLP[i][0]*self.maxXBeforeNormalizing))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(depthImg,coord, 1, (255,255,255), -1)
            cv2.putText(depthImg, "{}".format(text),coord, font, 0.3,(255,255,255),
                        1,cv2.LINE_AA)
        return depthImg

    def drawTrainDiff(self, img, calculatedTrainDepths, scaleFactor = 0.001):
        
        """
        Draws the points and the diff between target test set and the given
        calculated test set.

        Args:
            img([[Double]])                 : The image that should be drawn on
            calculatedTrainDepths([Double]) : The depths calculated by the MLP
            scaleFactor(float)              : The depths will be scaled by this 
                                              factor

        Returns([[Double]]):
            The image with the differences drawn
        """
        
        depthImg = img.copy()

        denormTarget = self.denormalizeTarget(self.targetTestMLP)
        denormCalculated = self.denormalizeTarget(calculatedTrainDepths)
        diff = (denormTarget - denormCalculated)*scaleFactor

        for i in range(0,len(self.targetTestMLP)):
            
            text = "{0:.1f}".format(diff[i][0])

            coord = (int(self.inputTestMLP[i][1]*self.maxYBeforeNormalizing), 
                     int(self.inputTestMLP[i][0]*self.maxXBeforeNormalizing))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(depthImg,coord, 1, (255,255,255), -1)
            cv2.putText(depthImg, text,coord, font, 0.3,(0,0,0),
                        1,cv2.LINE_AA)

        
        return depthImg

if __name__ == '__main__':  

    #Define image paths
    imgLeftPath = "../Images/CityLeft1.png"
    imgRightPath = "../Images/CityRight1.png"
    picklePath = "../Pickle/City1.p"

    #usedSegments = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    usedSegments = [2,3,4,5,6,7,8,9,10,11,12] # City1

    activationFuncs = [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid]

    optimizerFuncs = [tf.train.AdamOptimizer(0.1),tf.train.AdamOptimizer(0.01),
                 tf.train.AdamOptimizer(0.001),tf.train.AdamOptimizer(0.0001)]

    lossFunc = tf.losses.absolute_difference

    #Create the mlp data
    mlpDataProsessor = MLPDataProsessor(imgLeftPath, imgRightPath,
                                        usedSegments=usedSegments, 
                                        picklePath=picklePath)

    mlpDataProsessor.normalizeMLPData()
    mlpDataProsessor.splitTest(0.1)
    inputMLP, targetMLP, inputTestMLP, targetTestMLP = mlpDataProsessor.getMLPData()
    mlp = MLP.MLP(len(inputMLP[0]),1,[15,15],optimizerFuncs, activationFuncs, 
                  lossFunc)
    
    #Train the mlp and format the data
    mlp.train(inputMLP, targetMLP, batchSize=100, earlyStopEpochs=3,iterations=75, 
              maxEpochs=500, verbose=True, kFoldValidationSize=0.20, epochsBetweenPrint=10)
        
    array = mlpDataProsessor.convertSegmentCoordinatesToArray(usedSegments)
    result = mlp.forward(array)
    resultDenormalized = mlpDataProsessor.denormalizeTarget(result)
    resultImageNonLog = mlpDataProsessor.convertResultArrayToImage(result, usedSegments)
    resultImage = mlpDataProsessor.convertResultArrayToImage(resultDenormalized, usedSegments)

    #Print/ show the result
    maxDepth = resultImage.max()
    print("Maximum calculated distance from training data = {}m".format(maxDepth*0.001))
    print("Minimum calculated distance from training data = {}m".format(resultImage.min()*0.001))

    mlpDataInterpreter = MLPDataInterpreter.MLPDataInterpreter()
    testResults = mlp.forward(inputTestMLP)
    resultImage = mlpDataInterpreter.applyLogJetColorMap(resultImage)
    resultImage = mlpDataInterpreter.createColorBarWithLogDepth(maxDepth, resultImage)
    depthImage = mlpDataProsessor.drawTargets(resultImage)
    diffImage = mlpDataProsessor.drawTrainDiff(resultImage, testResults)

    testResultDenorm = mlpDataProsessor.denormalizeTarget(testResults)
    testTrueDenorm = mlpDataProsessor.denormalizeTarget(targetTestMLP)
    print("Mean testset diff: {}mm".format(np.mean(abs(testResultDenorm-testTrueDenorm))))

    cv2.imshow("Result image", resultImage)
    cv2.imshow("Result image with training depths", depthImage)
    cv2.imshow("Result image with test set difference", diffImage)
    cv2.imwrite("../Results/Resultimage.png", resultImage)
    cv2.imwrite("../Results/ResultimageWithTraining.png", depthImage)
    cv2.imwrite("../Results/ResultimageWithSparseDiffs.png", diffImage)
    cv2.waitKey(0)