import sys
sys.path.insert(0, './src')

import numpy as np
import cv2
import sys
import pickle
from Watershed import *

watershedColor = Watershed.loadSegmentedImg(picklePath="../Pickle/City1.p")
watershedColor = watershedColor/watershedColor.max()
watershedColor *= 255
watershedColor = watershedColor.astype(np.uint8)
watershedColor = cv2.applyColorMap(watershedColor, cv2.COLORMAP_JET)

cv2.imshow("Segmented Image", watershedColor)
cv2.imwrite("../Results/Segmented.png", watershedColor)
cv2.waitKey(0)