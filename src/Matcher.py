import numpy as np
import cv2
import time

class Matcher():

    """
    A class for computing features, keypoints and matches between stereo images
    
    Class Parameters:
        keypointsLeft([KeyPoint])   : The keypoints in the left image
        keypointsRight([KeyPoint])  : The keypoints in the right image
        descriptorsLeft([[float]])  : The descriptors of the left keypoints
        descriptorsRight([[float]]) : The descriptors of the right keypoints
        matches([Dmatch, Dmatch])   : The best and second best matches
        goodMatches([Dmatch])       : Matches after ratio and epipolar test

    Args:
        imgLeftPath (String)    : The path of the left stereo image file
        imgRightPath(String)    : The path of the right stereo image file
        featureFunc (Function)  : The function used for feature calculation
        descriptorFunc(Function): The function used for descriptor 
                                  calculation
        matcherFunc(Function)   : The function used to match keypoints
        epipolarTol(Int)        : The max pixel tolerance for two points 
                                  to be considered as on the same epipolar line
    """

    def __init__(self, imgLeft, imgRight, featureFunc=None, 
                 descriptorFunc=None, matcherFunc=None, epipolarTol = 0.7,
                 ratioThres=0.8):
        
        self.featureFunc = featureFunc
        self.descriptorFunc = descriptorFunc
        self.matcherFunc = matcherFunc
        self.epipolarTol = epipolarTol
        self.ratioThres = ratioThres

        self.imgLeft = imgLeft 
        self.imgRight = imgRight 

         #Assert that images exist
        assert len(np.shape(self.imgLeft)) > 0, "Left image was empty"
        assert len(np.shape(self.imgRight)) > 0, "Right image was empty"

        self.grayRight = cv2.cvtColor(self.imgRight,cv2.COLOR_BGR2GRAY)
        self.grayLeft = cv2.cvtColor(self.imgLeft, cv2.COLOR_BGR2GRAY)

        #Create standard functions if not given
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000000, contrastThreshold=0.003,
                                           edgeThreshold = 20)

        if(self.featureFunc == None):
            self.featureFunc = sift.detect
        if(self.descriptorFunc == None):
            self.descriptorFunc = sift.compute
        if(self.matcherFunc == None):
            self.matcherFunc = SGBFMatcher(imgLeft.shape)

        self.keypointsLeft = None
        self.keypointsRight = None
        self.descriptorsLeft = None
        self.descriptorsRight = None
        self.matches = None
        self.goodMatches = None


    def computeAll(self):

        """
        Computes keypoints, descriptors and good matches
        """

        self.computeKeypoints()
        self.rowSortKeypoints() #Needed for our custom matching algorithm
        self.computeDescriptors()
        self.computeMatches()
        self.extractGoodMatches()

    def rowSortKeypoints(self):

        """
        Sorts keypoints per row instead of column
        """

        self.keypointsLeft = sorted(self.keypointsLeft, key=lambda x: x.pt[1])
        self.keypointsRight = sorted(self.keypointsRight, key=lambda x: x.pt[1])

    def computeKeypoints(self):

        """
        Computes keypoints using self.featureFunc
        """

        self.keypointsLeft = self.featureFunc(self.grayLeft,None)
        self.keypointsRight = self.featureFunc(self.grayRight,None)

    def computeDescriptors(self):

        """
        Computes descriptors using self.descriptorFunc
        """

        msg = "ERROR: no keypoints found, run computeKeypoints before "\
              "computeDescriptors"
        assert len(np.shape(self.keypointsLeft)) > 0, msg 

        self.keypointsLeft, self.descriptorsLeft = self.descriptorFunc(self.imgLeft, 
                                                       self.keypointsLeft, 
                                                       None)
        self.keypointsRight, self.descriptorsRight = self.descriptorFunc(self.imgRight, 
                                                        self.keypointsRight, 
                                                        None)
    def computeMatches(self):

        """
        Computes matches using self.matcherFunc
        """

        msg = "ERROR: no descriptors found, run computeDescriptors before "\
              "computeMatches"
        assert len(np.shape(self.descriptorsLeft)) > 0, msg


        """
        A crude hack to make the class work with opencv matchers as well as 
        our custom matcher which needs keypointsRight and keypointsleft
        """
        if(isinstance(self.matcherFunc, SGBFMatcher)):
            self.matches = self.matcherFunc(self.descriptorsLeft, 
                                            self.descriptorsRight,
                                            self.keypointsLeft,
                                            self.keypointsRight) 
        else: 
            self.matches = self.matcherFunc(self.descriptorsLeft, 
                                            self.descriptorsRight, k=2)  

    def extractGoodMatches(self):

        """
        Extracts matches that pass the ratio test and are on the same epipolar 
        line
        """

        msg = "ERROR: No matches found, run matches() before "\
              "extractGoodMatches"
        assert len(np.shape(self.matches)) > 0, msg
        self.goodMatches = []

        for m,n in self.matches:
            if n.queryIdx == -1:
                self.goodMatches.append([m])
                continue

            keyPtLeft = self.keypointsLeft[m.queryIdx]
            keyPtRight = self.keypointsRight[m.trainIdx]

            #Check for negative disparity
            leftX = keyPtLeft.pt[0]
            rightX = keyPtRight.pt[0]
            if(leftX - rightX <= 0): 
                continue

            #Check if angles match
            angleDiff = abs(keyPtLeft.angle - keyPtRight.angle)
            angleDiff = min(angleDiff, abs(keyPtLeft.angle - keyPtRight.angle))
            if(angleDiff > 10):
                continue

            #Check if scale matches
            sizeDiff = abs(keyPtLeft.size - keyPtRight.size)
            if(sizeDiff > 1):
                continue

            #Epipolar test
            leftY = keyPtLeft.pt[1]
            rightY = keyPtRight.pt[1]
            if(abs(leftY - rightY) > self.epipolarTol):
                continue

            #Ratio test
            if m.distance < self.ratioThres*n.distance:
                self.goodMatches.append([m])


    def getKeypointImages(self):

        """
        Returns the original images with keypoints marked

        Returns([[Double]], [[Double]]):
            keypointImgLeft: The left image with keypoints marked
            keypointImgRight: The right image with keypoints marked
        """

        keypointImgLeft = cv2.drawKeypoints(self.imgLeft, self.keypointsLeft, 
                                            None)
        keypointImgRight = cv2.drawKeypoints(self.imgRight, self.keypointsRight, 
                                             None)
        return keypointImgLeft, keypointImgRight

    def getMatchesImg(self):

        """
        Returns the left and right image with matches marked
        """

        matchesImg = cv2.drawMatchesKnn(self.imgLeft, self.keypointsLeft,
                                        self.imgRight, self.keypointsRight,
                                        self.goodMatches, None, flags=2)
        return matchesImg


class SGBFMatcher():

    """
    Stereo Geometry Brute Force Matcher

    Mathes features along epipolar lines  within a threshold and refines the 
    mathces to be spread out in all parts of the image.
    
    Class Parameters:
        matches([[DMatch]]): The matches created by __call__()

    Args:
        imgSize([Int, Int]) : The size of the image
        lowThres(float)     : All matches with distance below this are accepted
        highThres(float)    : Used by refineMatches() to accepted matches below 
                              this distance if few matches in the neighboorhod 
        epipolarTol(float)  : Accepted tolerance for the epipolar line test
        localSize(float)    : The size of the neigborhood used in refineMatches()
        localNum            : The best localNum matches in a neigborhood are 
                              accepted


    """

    def __init__(self, imgSize, lowThres=100, highThres=200, epipolarTol=5, 
                 localSize=15, localNum=4):

        self.imgSize = imgSize
        self.lowThres=lowThres
        self.highThres=highThres
        self.epipolarTol=epipolarTol
        self.localSize=localSize
        self.localNum=localNum


    def __call__(self, descriptorsLeft, descriptorsRight, keypointsLeft,
                keypointsRight):

        """
        Runs brute force matching and refining.

        Notes:
            Keypoints and descriptors have to be sorted along rows
        
        Args:
            descriptorsLeft([float]) : The left descriptors
            descriptorsRight([float]): The right descriptors
            keypointsLeft       : Features in the left image
            keypointsRight      : Features in the right image
        
        Returns([DMatch]):
            The matches
        """

        self.keypointsLeft = keypointsLeft
        self.keypointsRight = keypointsRight
        self._match(descriptorsLeft, descriptorsRight)
        self._refineMatches()
        return self.matches

    def _match(self, descriptorsLeft, descriptorsRight):
        
        """
        Brute force matches along the epipolar line +/- self.epipolarTolerance
        
        Args:
            descriptorsLeft([float]) : The left descriptors
            descriptorsRight([float]): The right descriptors
        """

        currentRight = 0
        matches = np.empty((len(descriptorsLeft)+1, 2), dtype=cv2.DMatch)
        currentMatch = 0
        matches[currentMatch][0] = cv2.DMatch()
        matches[currentMatch][1] = cv2.DMatch()
        numPointsLeft = len(descriptorsLeft)
        numPointsRight = len(descriptorsRight)

        for i in range(numPointsLeft):
            
            keyPtLeft = self.keypointsLeft[i]

            for j in range(currentRight, numPointsRight):
                keyPtRight = self.keypointsRight[j]

                #Check if point in right image is over epipolar tolerance
                if( keyPtLeft.pt[1] - keyPtRight.pt[1] > self.epipolarTol):
                    currentRight += 1
                    continue

                #Check if point in right image is belove epipolar tolerance
                if(keyPtLeft.pt[1] - keyPtRight.pt[1] < - self.epipolarTol):
                    break

                score = np.linalg.norm(descriptorsLeft[i] - descriptorsRight[j])
                
                if(score > self.highThres):
                    continue

                if( matches[currentMatch][0].queryIdx == -1 or 
                    score < matches[currentMatch][0].distance):

                    #Move best to second best match
                    matches[currentMatch][1].queryIdx = i
                    matches[currentMatch][1].trainIdx = matches[currentMatch][0].trainIdx
                    matches[currentMatch][1].distance = matches[currentMatch][0].distance

                    #Save new best match
                    matches[currentMatch][0].queryIdx = i
                    matches[currentMatch][0].trainIdx = j
                    matches[currentMatch][0].distance = score

                elif(matches[currentMatch][1].queryIdx == -1 or 
                     score < matches[currentMatch][1].distance):

                    #Save new second best match
                    matches[currentMatch][1].queryIdx = i
                    matches[currentMatch][1].trainIdx = j
                    matches[currentMatch][1].distance = score

            if  matches[currentMatch][0].queryIdx != -1:

                currentMatch += 1
                matches[currentMatch][0] = cv2.DMatch()
                matches[currentMatch][1] = cv2.DMatch()
            
        self.matches = matches[0:currentMatch]
    
    def _refineMatches(self):

        """
        Refines the matches to only acept strong matches in regions with many
        matches and weaker matches in other regions
    
        All matches with distance below self.lowThres are accepted. Other 
        matches are only accepted if they are among the best self.localNum matches
        in a neigborhood of [self.localSize, self.localSize]    
        """

        matchMap = self._createMatchMap()
        newMatches = []

        for match in self.matches:
            if(match[0].distance < self.lowThres):
                newMatches.append(match)
            
            pt = self.keypointsLeft[match[0].queryIdx].pt 
            closeMatches = self._getLocalMatches(matchMap, [pt[1], pt[0]])

            if(len(closeMatches) < self.localNum +1 or match[0].distance <= 
                self.matches[closeMatches[self.localNum]][0].distance):
                newMatches.append(match)

        self.matches = newMatches
        
    def _getLocalMatches(self, matchMap, pt):
        
        """
        Returns the matches in a neigboorhood of size: 
        [self.localSize, self.localSize]
        
        Args:
            localMatchMap([[Int]])  : A map of the same size as the image where 
                                      the match indicies in self.matches array 
                                      are stored
            pt([float, float])      : The coordinates of the center of the 
                                       neighborhood 
        """
        
        imgMaxX = self.imgSize[0]
        imgMaxY = self.imgSize[1]

        minX = int(pt[0] - self.localSize//2)
        if minX < 0:
            minX = 0

        maxX = int(pt[0] + self.localSize//2)
        if maxX > imgMaxX:
            maxX = imgMaxX

        minY = int(pt[1] - self.localSize//2)
        if minY < 0:
            minY = 0
            
        maxY = int(pt[1] + self.localSize//2)
        if maxY > imgMaxY:
            maxY = imgMaxY

        
        localMatchMap = matchMap[minX:maxX, minY:maxY]

        indicies = localMatchMap[localMatchMap != 0]  
        return sorted(indicies, key=lambda x: self.matches[x][0].distance) 
        
    
    def _createMatchMap(self):

        """
        creates A map of the same size as the image where the match indicies 
        in self.matches array are stored at the coorect pixel coordinates
        """ 

        matchMap = np.zeros(self.imgSize, dtype=np.int32)
        
        for i in range(len(self.matches)):
            pt = self.keypointsLeft[self.matches[i][0].queryIdx].pt
            matchMap[int(pt[1])][int(pt[0])] = i

        return matchMap

if __name__ == "__main__":

    """ Example run """

    #Load images
    imgLeft = cv2.imread('../Images/CityLeft1.png')
    imgRight = cv2.imread('../Images/CityRight1.png')

    #Create the functions that should be used in Matcher
    # (Not necessary as standard functions are implmented)
    algo = cv2.xfeatures2d.SIFT_create(nfeatures = 100000, edgeThreshold=20,
                                       contrastThreshold=0.003)
    featureFunc = algo.detect
    descriptorFunc = algo.compute
    matcherFunc = SGBFMatcher(imgLeft.shape)

    #Create the matcher object and compute features, descriptors and matches
    matcher = Matcher(imgLeft, imgRight, featureFunc, descriptorFunc, matcherFunc)
    matcher.computeAll()

    #Display result
    keypointImgLeft, keypointImgRight = matcher.getKeypointImages()
    matchesImg = matcher.getMatchesImg()
    cv2.imshow('Keypoints Right',keypointImgRight)
    cv2.imshow('Keypoints Left',keypointImgLeft)
    cv2.imshow('Matches',matchesImg)

    cv2.waitKey(0)
