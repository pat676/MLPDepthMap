import numpy as np
import cv2
import Matcher

class StereoDepth():

    """
    A class for computing the depth of stereo matches

    Args:
        matches(cv2.Dmatch)         : The matches for disparity/depth calculation
        keypointsLeft(cv2.KeyPoint) : Keypoints of the left image
        keypointsRight(cv2.KeyPoint): Keypoints for the right image
        f(float)                    : The focal length
        bx(float)                   : The baseline
        doffs(float)                : C1x - C0x

    Class Variables:
        disparities(Double): Disparities of the matches
        depths(Double): The calculated depths of the matches
    """
    
    def __init__(self, matches, keypointsLeft, keypointsRight, f, bx, doffs=0):
        self.matches = matches
        self.keypointsLeft = keypointsLeft
        self.keypointsRight = keypointsRight

        self.f = f
        self.bx = bx
        self.doffs = doffs
        self.disparities = []
        self.depths = []

    def computeAll(self):

        """
        Computes disparities, calibration parameters and depths
        """

        self.computeDisparities()
        self.computeDepths()

    def computeDisparities(self):

        """
        Uses the matches to compute disparities
        """

        self.disparities = []
        for match in self.matches:
            ptLeft = self.keypointsLeft[match[0].queryIdx].pt
            ptRight = self.keypointsRight[match[0].trainIdx].pt

            disparity = ptLeft[0] - ptRight[0]
            self.disparities.append(disparity)

    def computeDepths(self):

        """
        Uses f, bx and goodMatches to compute depths
        """

        msg = "ERROR: No disparities found, run computeDisparities before "\
               "computeDepths"
        assert len(self.disparities) > 0, msg

        if(self.bx == 0 or self.f  == 0):
            print("Error computing depths. bx and f can't be zero."+ 
                  "Be sure to run extractCalibs before running computeDepts()")
            return False

        fbx = self.f * self.bx
        for i in range(0,len(self.disparities)):
            self.depths.append(fbx/(self.disparities[i]+self.doffs))

    def getDepthImage(self, img, depthScale=0.001):

        """
        Creates depth markers on the given image

        Args:
            img([[int]])     : The original image
            depthScale(float): The depths are scaled by this amount

        Returns img([[int]]):
            The image with depth markers
        """

        depthImg = img.copy()

        for i in range(0,len(self.depths)):

            depth = self.depths[i]
            text = "{0:.2f}".format(depth*depthScale)
            pos = self.keypointsLeft[self.matches[i][0].queryIdx].pt
            coord = (int(pos[0]), int(pos[1]))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(depthImg,coord, 1, (255,0,0), -1)
            cv2.putText(depthImg, "{}".format(text),coord, font, 0.3,(0,0,255),
                        1,cv2.LINE_AA)

        return depthImg


if __name__ == '__main__':

    #Load images
    #imgLeft = cv2.imread('../Images/CityLeft1.png')
    #imgRight = cv2.imread('../Images/CityRight1.png')

    imgLeft = cv2.imread('../Images/middelburyLeft.png')
    imgRight = cv2.imread('../Images/middelburyRight.png')

    #Create the functions that should be used in Matcher
    matcher = Matcher.Matcher(imgLeft, imgRight)
    matcher.computeAll()

    #f and bx from the kitty dataset
    f = 7.215377e+02
    bx = 4.485728e+01 + 3.395242e+02
    
    sd = StereoDepth(matcher.goodMatches, matcher.keypointsLeft, 
                     matcher.keypointsRight, f, bx)
    sd.computeAll()

    img = sd.getDepthImage(matcher.imgLeft)

    cv2.imshow("Original image", matcher.imgLeft)
    cv2.imshow("Depth image", img)
    cv2.imwrite("../Results/StereoDepth.png", img)
    print("Number of good matches: {}".format(len(matcher.goodMatches)))
    cv2.waitKey(0)





        