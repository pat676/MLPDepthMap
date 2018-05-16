import sys
sys.path.insert(0, './src')

import numpy as np 
import cv2
import StereoDepth
import Matcher
import Watershed
import tensorflow as tf
import MLP
import MLPDataProsessor
import MLPDataInterpreter

class AR:

	"""
	Class (description):
		A class used to calculate depths in a scene image and object image and 
		placing the images in the same result image.

		The object image can be given a depth offest to place it behind objects 
		in the scene image. 

		If imgPathRO is 0, then depth of object is not calculated, instead a
		constant depth is assumed for the object, this enables the user to add any
		type of object (not restricted to objects segmented from stereo camera 
		images). The object is given through picklePathO and imgPathLO

	Args:
		picklePath(String)	 	: Path to pickle file of segmented scene image
		picklePathO(String)  	: Path to pickle file with segmented object 
								  image
		imgPathL(String)	 	: Image path to left scene image in stereo 
								  camera model
		imgPathR(String)	 	: Image path to right scene image in stereo 
								  camera model
		segments(String)	 	: segments to calculate depths on, in the 
							      scene image	
		imgPathLO(String)	 	: Image path to left object image in stereo 
							      camera model
		imgPathRO(String)	 	: Image path to right object image in stereo 
								  camera model
		hiddenLayers([int])		: hidden layers for MLP scene image
		activationFuncsO      	: activation functions for MLP scene image
		  (Tensorflow.nn.'func') 
		hiddenLayersO([int]) 	: hidden layers for MLP object image	  
		activationFuncsO		: activation functions for MLP object image
		  (Tensorflow.nn.'func')
		"""

	def __init__(self, picklePath, picklePathO, imgPathL, imgPathR, segments, 
				 imgPathLO, imgPathRO=0, hiddenLayers=[15,20,15], hiddenLayersO=[4,2],
				 activationFuncs=[tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid],
				 activationFuncsO=[tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid]):

		self.picklePath, self.picklePathO = picklePath, picklePathO
		self.imgPathL, self.imgPathR = imgPathL, imgPathR
		self.imgPathLO, self.imgPathRO = imgPathLO, imgPathRO

		self.segmentedObjectImage = Watershed.Watershed.loadSegmentedImg(self.picklePathO)
		self.objectImage = cv2.imread(self.imgPathLO)
		
		if(self.imgPathRO != 0):
			self.depthObjectImage = self._depthOfObject(hiddenLayersO, activationFuncsO)

		else:
			#depth image is basically the same as segmented image, this will
			#make the depth constant
			self.depthObjectImage = self._formatSegmentImg(self.segmentedObjectImage, 2)

		self._cropAndCenterObjectImages()
		self._peelObjectImg()

		self.sceneImage = cv2.imread(self.imgPathL)
		self.depthSceneImage = self._depthOfScene(hiddenLayers, activationFuncs, segments)



	def _depthOfObject(self, hiddenLayersO, activationFuncsO, segmentNum=2):

		"""
		Calculate depth of object using our method.

		Args:
			hiddenLayersO([int])	: hidden layers for MLP calculating the depths
			activationFuncsO    : activation functions for MLP scene image
		  	  (Tensorflow.nn.'func') 
		  	segmentNum(int)		: The segment number of the object
		"""

		usedSegments = [segmentNum]

		optimizerFuncs = [tf.train.AdamOptimizer(0.1),tf.train.AdamOptimizer(0.01),
		                 tf.train.AdamOptimizer(0.001),tf.train.AdamOptimizer(0.0001)]
		lossFunc = tf.losses.absolute_difference

		#Create the mlp data
		mlpDataProsessor = MLPDataProsessor.MLPDataProsessor(self.imgPathLO, self.imgPathRO,
															 usedSegments=usedSegments, 
															 picklePath=self.picklePathO)
		mlpDataProsessor.normalizeMLPData()
		inputMLP, targetMLP, inputTestMLP, targetTestMLP = mlpDataProsessor.getMLPData()
		mlp = MLP.MLP(len(inputMLP[0]),1,hiddenLayersO,optimizerFuncs, activationFuncsO, 
						  lossFunc)


		#Train the mlp and format the data
		mlp.train(inputMLP, targetMLP, batchSize=100, earlyStopEpochs=3,iterations=75, maxEpochs=2000, 
				  verbose=True, kFoldValidationSize=0.20, epochsBetweenPrint=10)
		array = mlpDataProsessor.convertSegmentCoordinatesToArray(usedSegments)
		result = mlp.forward(array)
		resultDenormalized = mlpDataProsessor.denormalizeTarget(result)
		resultImage = mlpDataProsessor.convertResultArrayToImage(resultDenormalized, usedSegments)

		return resultImage

	def _depthOfScene(self, hiddenLayers, activationFuncs, segments):

		"""
		Calculate depth of scene using our method. segments to calculate depths on 
		has to be chosen with belonging pickleFile

		Args:
			hiddenLayers([int])	: hidden layers for MLP calculating the depths
			activationFuncs     : activation functions for MLP scene image
		  	  (Tensorflow.nn.'func') 
		  	segmentNum(int)		: The segments to be used in the depth calculations
		"""
		
		usedSegments = segments

		optimizerFuncs = [tf.train.AdamOptimizer(0.1),tf.train.AdamOptimizer(0.01),
		                 tf.train.AdamOptimizer(0.001),tf.train.AdamOptimizer(0.0001)]
		lossFunc = tf.losses.absolute_difference

		#Create the mlp data
		mlpDataProsessor = MLPDataProsessor.MLPDataProsessor(self.imgPathL, self.imgPathR,
															 usedSegments=usedSegments, 
															 picklePath=self.picklePath)
		mlpDataProsessor.normalizeMLPData()
		inputMLP, targetMLP, inputTestMLP, targetTestMLP = mlpDataProsessor.getMLPData()
		mlp = MLP.MLP(len(inputMLP[0]),1,hiddenLayers,optimizerFuncs, activationFuncs, 
						  lossFunc)


		#Train the mlp and format the data
		mlp.train(inputMLP, targetMLP, batchSize=100, earlyStopEpochs=3,iterations=75, maxEpochs=2000, 
				  verbose=True, kFoldValidationSize=0.20, epochsBetweenPrint=10)
		array = mlpDataProsessor.convertSegmentCoordinatesToArray(usedSegments)
		result = mlp.forward(array)
		resultDenormalized = mlpDataProsessor.denormalizeTarget(result)
		resultImage = mlpDataProsessor.convertResultArrayToImage(resultDenormalized, usedSegments)


		resultImage = self._interpolateEdges(resultImage)
		#make sky 10 times max depth
		resultImage[resultImage==0] = resultImage.max()*10
		return resultImage

	def _interpolateEdges(self, depthImg):

		"""
		Give edges max of nearest neighbour depth (for completeness) Ignoring the 
		border of the scene image as it won't affect the result at a particulary
		high degree

		Args:
			depthImg([[Double]]): The depth image
		"""
		
		rows, cols = depthImg.shape
		depthImgCpy = depthImg.copy()

		for i in range(1, rows-1, 1):
			for j in range(1, cols-1, 1):
				if(depthImg[i,j] == 0):
					depthImgCpy[i,j] = depthImg[i-1:i+2:1, j-1:j+2:1].max()

		return depthImgCpy

	def _formatSegmentImg(self, im, segmentNum):

		"""
		Helper method for formatting a segmented image.
		Set background = 0 and objct = 1 in img

		Args:
			im([[Double]])	: The image that should be formated
			segmentNum(int)	: The number of the object segment
		
		"""

		img = im.copy()
		img[img != segmentNum] = 0
		img = img/segmentNum
		return img

	def _cropAndCenterObjectImages(self, segmentNum=2):

		"""
		Crops the image containing our wanted object to minimal size in height and width
		which can still handle any degree of rotation without loosing part of object out
		of bounds (done by padding). This is done to all three object images; 
			color(raw), depth, segmented

		Args:
			segmentNum(int)	: The segment number of the object
		"""

		numRows,numCols = self.segmentedObjectImage.shape
		self.segmentedObjectImage = self._formatSegmentImg(self.segmentedObjectImage, segmentNum)

		searchMinHeight = True
		for i in range(numRows):

			#pick lowest pixel index in height
			if(self.segmentedObjectImage[i].max() == 1 and searchMinHeight):
				imgHeightMin = i
				searchMinHeight = False 

			#pick highest pixel index in height
			if(self.segmentedObjectImage[i].max() == 1 and not searchMinHeight):
				imgHeightMax = i

		searchMinWidth = True
		for i in range(numCols):

			if(self.segmentedObjectImage[:, i].max() == 1 and searchMinWidth):
				imgWidthMin = i
				searchMinWidth = False

			if(self.segmentedObjectImage[:, i].max() == 1 and not searchMinWidth):
				imgWidthMax = i

		self.croppedObjectImg = self.objectImage[imgHeightMin:imgHeightMax:1, 
												   imgWidthMin:imgWidthMax:1, :].copy()
		self.croppedSegmentedObjectImg = self.segmentedObjectImage[imgHeightMin:imgHeightMax:1, 
																   imgWidthMin:imgWidthMax:1].copy()
		self.croppedDepthObjectImg = self.depthObjectImage[imgHeightMin:imgHeightMax:1, 
														   imgWidthMin:imgWidthMax:1].copy()

		objectHeight = imgHeightMax - imgHeightMin
		objectWidth = imgWidthMax - imgWidthMin

		if(objectHeight < objectWidth):
			pad1 = int((objectWidth - objectHeight)/2)
			pad2 = pad1 + (objectWidth - objectHeight)%2
			pad11 = 0
			pad22 = 0

		else:
			pad1, pad2 = 0, 0
			pad11 = int((objectHeight - objectWidth)/2)
			pad22 = pad11 + (objectHeight - objectWidth)%2

		self.croppedSegmentedObjectImg = np.pad(self.croppedSegmentedObjectImg, 
												((pad1, pad2), (pad11,pad22)), 
												mode='constant', constant_values=0)
		self.croppedDepthObjectImg = np.pad(self.croppedDepthObjectImg, ((pad1, pad2), 
											(pad11,pad22)), mode='constant', 
											 constant_values=0)
			
		bgr = []
		for color in cv2.split(self.croppedObjectImg):
			bgr.append(np.pad(color, ((pad1, pad2), (pad11,pad22)), mode='constant', constant_values=0))

		self.croppedObjectImg = cv2.merge((bgr[0],bgr[1],bgr[2]))

	def _peelObjectImg(self):

		"""
		Only the object which has been segmented is to be left in the cropped color object
		image. Peel the rest of the non-object.
		"""

		self.croppedObjectImg[self.croppedSegmentedObjectImg!=1] = (0,0,0)

	def _rotateImgs(self, imgs, theta):

		"""
		rotate images by theta

		Args:
			imgs([[Double]]): element1: croppedSegmentedObjectImage
							  element2: croppedDepthObjectImage
							  element3: croppedObjectImage
			theta(float)	: rotational parameter
		"""

		rows, cols = imgs[0].shape
		M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, 1)
		
		rotCOIs = []	
		for i in range(len(imgs)):
			rotCOIs.append(cv2.warpAffine(imgs[i], M, (cols, rows)))

		return rotCOIs

	def _scaleImgs(self, imgs, scaleX, scaleY):

		"""
		scale images by scale factors

		Note:
			better results depend on better interpolation, but this is computationally more expensive
			as well as we have to deal with downsizing and upsizing separately
		
		Args:
			imgs([[Double]]): element1: croppedSegmentedObjectImage
							  element2: croppedDepthObjectImage
							  element3: croppedObjectImage
			ScaleX(float)	: scale factor in x-direction
			scaleY(float)	: scale factor in y-direction
		"""

		scaleCOIs = []
		for i in range(len(imgs)):
			scaleCOIs.append(cv2.resize(imgs[i], None, fx=abs(scaleX), fy=abs(scaleY)))

			if(scaleX < 0):
				scaleCOIs[i] = cv2.flip(scaleCOIs[i], 1)

			if(scaleY < 0):
				scaleCOIs[i] = cv2.flip(scaleCOIs[i], 0)

		return scaleCOIs

	def _putObjectInScene(self, imgs, x, y, depthOffset=0):

		"""
		Place object image in the scene image. Only the pixels that are at a lower 
		depth than those in the scene though.

		Args:
			imgs([[Double]]): element1: croppedSegmentedObjectImage
							  element2: croppedDepthObjectImage
							  element3: croppedObjectImage
			x, y(int)		: scene image coordinates corresponding to object 
							  image center
			depthOffset 	: depth offset to place object at in scene image
		"""

		cpySceneImage = self.sceneImage.copy()

		x0 = x - int(imgs[0].shape[0]/2)
		y0 = y - int(imgs[0].shape[1]/2)

		index = np.argwhere(imgs[0]==1)
		index += (x0, y0)

		for idx in index:

			if(idx.min() < 0):
				continue

			elif(idx[0] >= cpySceneImage.shape[0] or idx[1] >= cpySceneImage.shape[1]):
				continue

			elif(self.depthSceneImage[idx[0], idx[1]] >= imgs[1][idx[0]-x0, idx[1]-y0]):
				cpySceneImage[idx[0], idx[1], :] = imgs[2][idx[0]-x0, idx[1]-y0, :]

		return cpySceneImage

	def _saveObjectInScene(self, imgs, x, y, depthOffset=0):

		"""
		Save object image in the scene image. Only the pixels that are at a lower 
		depth than those in the scene though.

		Args:
			imgs([[Double]]): element1: croppedSegmentedObjectImage
							  element2: croppedDepthObjectImage
							  element3: croppedObjectImage
			x, y(int)		: scene image coordinates corresponding to object 
							  image center
			depthOffset 	: depth offset to place object at in scene image
		"""

		x0 = x - int(imgs[0].shape[0]/2)
		y0 = y - int(imgs[0].shape[1]/2)

		index = np.argwhere(imgs[0]==1)
		index += (x0, y0)

		for idx in index:

			if(idx.min() < 0):
				continue

			elif(idx[0] >= self.sceneImage.shape[0] or idx[1] >= self.sceneImage.shape[1]):
				continue

			elif(self.depthSceneImage[idx[0], idx[1]] >= imgs[1][idx[0]-x0, idx[1]-y0]):
				self.sceneImage[idx[0], idx[1], :] = imgs[2][idx[0]-x0, idx[1]-y0, :]
				self.depthSceneImage[idx[0], idx[1]] = imgs[1][idx[0]-x0, idx[1]-y0]

	def _normalizeObjectDepthAndAddOffset(self, depthImage, offset):

		"""
		Make lowest depth = 0, then add offset. Make background depth not be relevant

		Args:
			depthImage([[Double]])	: image with values corresponding to depth 
									  calculations
			Offset(float)			: the depth offset to add to the object
		"""

		backgroundDepth = 10000000
		depthImage[depthImage==0] = backgroundDepth
		depthImage[depthImage!=backgroundDepth] -= depthImage[depthImage!=backgroundDepth].min() - offset

		return depthImage

	def getImages(self, x=0, y=0, theta=0, scaleX=1, scaleY=1, depthOffset=0):
		
		"""
		Create imgs for easy handling of object images, for looping and such.

		Reshapes and rotates the object image, and then places it into the scene
		in AR fashion.

		Note:
			This is the only methods that should be used (other methods are private).

		Args:
			x, y(Int)			: scene image coordinates corresponding to object 
						      	  image center
			theta(float)		: rotational degree of object image
			scaleX(float)		: scales the object image by factor scaleX in 
							  	  the x-direction
			scaleY(float)		: scales the object image by factor scaleY in 
							  	  the y-direction
			depthOffset(float)	: at which depth the image is to appear in the 
								  scene

		Returns([[Double]]):
			scene image with object and scaled and rotated object
		"""

		imgs = [self.croppedSegmentedObjectImg, self.croppedDepthObjectImg.copy(),
				self.croppedObjectImg]

		imgs[1] = self._normalizeObjectDepthAndAddOffset(imgs[1], depthOffset)
		rotCOIs = self._rotateImgs(imgs, theta)
		scaleCOIs = self._scaleImgs(rotCOIs, scaleX, scaleY)

		ARimg = self._putObjectInScene(scaleCOIs, x, y, depthOffset)

		#make background white for displayed object image
		scaleCOIs[2][scaleCOIs[0]==0] = (255,255,255)

		return [ARimg, scaleCOIs[2]]

	def saveCurrentImage(self, x=0, y=0, theta=0, scaleX=1, scaleY=1, depthOffset=0):
		
		"""
		Saves the current scene image with object as the new original scene image

		Args:
			x, y: scene image coordinates corresponding to object image center
			theta: rotational degree of object image
			scaleX: scales the object image by factor scaleX in the x-direction
			scaleY: scales the object image by factor scaleY in the y-direction
			depthOffset: at which depth the image is to appear in the scene
		"""

		imgs = [self.croppedSegmentedObjectImg, self.croppedDepthObjectImg.copy(),
				self.croppedObjectImg]

		imgs[1] = self._normalizeObjectDepthAndAddOffset(imgs[1], depthOffset)
		rotCOIs = self._rotateImgs(imgs, theta)
		scaleCOIs = self._scaleImgs(rotCOIs, scaleX, scaleY)
		self._saveObjectInScene(scaleCOIs, x, y, depthOffset)


	def showImgs(self, ARimg, img):

		"""
		Display AR image and cropped, scaled, rotated, color, object image.
		"""

		cv2.imshow("Cropped Object Image", img)
		cv2.imshow("AR image", ARimg)
		cv2.waitKey(0)



if __name__ == "__main__":

	ARObject = AR("../Pickle/City1.p", "../Pickle/pickleAnders.p",
					    "../Images/CityLeft1.png", "../Images/CityRight1.png", 
					    [2,3,4,5,6,7,8,9,10,11,12],"../Images/Anders.jpg")

	imgs = ARObject.getImages(x=100, y=750, theta=45, scaleX=0.25, scaleY=0.25, depthOffset=40)
	ARObject.showImgs(imgs[0],imgs[1])

