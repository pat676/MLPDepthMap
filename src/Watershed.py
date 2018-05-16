import numpy as np
import cv2
import sys
import PixelSelectGUI as PS
import pickle


class Watershed:

    """
    A class for image segmentation using opencv's Watershed implementation
    and manual marker selection using the PixelSelectGUI.

    Args:
        imgPath: Path to image that is being segmented
    """

    def __init__(self, imgPath):

        self.imgPath = imgPath
        self.img = cv2.imread(imgPath)
        msg = "Could not load image: {}".format(imgPath)
        assert len(np.shape(self.img)) > 0, msg

        self.imgShape = self.img.shape[0:2]
        self.ps = PS.PixelSelect(self.imgPath)

        #Number of segments after segment() has run
        self.numSegments = 0

    def segment(self):

        """
        Runs the PixelSelectGUI for manual marker selection. After selction
        the image is segmented using opencv's Watershed implementation

        Returns([[Double]]):
            The segmented image
        """

        #Run the marker selection GUI
        self.ps.startGUI()
        self.numSegments = self.ps.numSegments
        markerPoints = self.ps.result
        if(markerPoints == 0):
            print("No markers, exiting watershed...")
            return False

        markers = np.zeros(self.imgShape, dtype = np.uint8)
        
        #Format the markers to matrix
        for i in range(0,len(markerPoints)):
            for j in range(0,len(markerPoints[i])):
                x = markerPoints[i][j][0]
                y = markerPoints[i][j][1]

                markers[x,y] = (i+1)

        watershed = markers.copy().astype(np.int32)
        self.segmentedImg = cv2.watershed(self.img,watershed)
        return self.segmentedImg

    def segmentAndSaveImg(self, picklePath):

        """
        Save a segmented image
        """

        segment = self.segment()
        pickle.dump(segment, open(picklePath, "wb"))
        return segment

    @staticmethod
    def loadSegmentedImg(picklePath):

        """
        Load a saved segmented image
        """

        return pickle.load(open(picklePath, "rb"))


if __name__ == "__main__":

    imgPath = '../Images/CityLeft2.png'
    img = cv2.imread(imgPath)
    result = img.copy()

    wShed = Watershed(imgPath)

    #segment, save and load saved segmented image.
    wShed.segmentAndSaveImg(picklePath='../Pickle/Watershed_ex.p')
    watershed = wShed.segmentedImg
    result[watershed == -1] = [255,0,0]

    watershedColor = watershed.copy()
    watershedColor = watershedColor/watershedColor.max()
    watershedColor *= 255
    watershedColor = watershedColor.astype(np.uint8)
    watershedColor = cv2.applyColorMap(watershedColor, cv2.COLORMAP_JET)

    #cv2.imshow("Watershed", watershed/watershed.max())
    cv2.imshow("Result", result)
    cv2.imshow("jetMap", watershedColor)
    cv2.waitKey(0)






