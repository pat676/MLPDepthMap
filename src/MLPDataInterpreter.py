import numpy as np 
import cv2
import StereoDepth
import Matcher
import Watershed
import tensorflow as tf
import MLP
import MLPDataProsessor

class MLPDataInterpreter():

    """
    A class used to run statistics and visualize results of the depth map
    """

    def runStatistics(self, layers, activationFuncs, usedSegments,
                      imgLeftPath, imgRightPath, picklePath, MLPIterations=20,
                      scaleFactor=0.001):
        
        """
        Does a statistics run and prints the result to terminal

        Args:
            layers([Int])           : The number of nodes in each layer of the MLP
            activationFuncs([tf."activationFunc"]): activation funcs of the MLP
            usedSegments([int])     : The segments that should be used
            MLPIterations(Int)      : The number of iterations done to calculate 
                                      statistics
            imgLeftPath(String)     : The path of the left image
            imgRightPath(String)    : The path of the right image
            picklePath(String)      : The path of the pickle file containing the 
                                      segmentation       
        """


        optimizerFuncs = [tf.train.AdamOptimizer(0.1),tf.train.AdamOptimizer(0.01),
                 tf.train.AdamOptimizer(0.001),tf.train.AdamOptimizer(0.0001)]
        lossFunc = tf.losses.absolute_difference

        trainSetAvgDiff = np.zeros(MLPIterations)
        trainSetStd = np.zeros(MLPIterations)

        for i in range(MLPIterations):
            print("Running iteration: {}".format(i+1))

            mlpDataProsessor = MLPDataProsessor.MLPDataProsessor(imgLeftPath, imgRightPath, 
                                                                 usedSegments=usedSegments,
                                                                 picklePath=picklePath,
                                                                 clipPercentile=0.90)
            mlpDataProsessor.normalizeMLPData()
            mlpDataProsessor.splitTest(0.1)

            inputMLP, targetMLP, inputTestMLP, targetTestMLP = mlpDataProsessor.getMLPData()

            mlp = MLP.MLP(len(inputMLP[0]),1,layers,optimizerFuncs, activationFuncs, lossFunc)
            mlp.train(inputMLP, targetMLP, batchSize=100, earlyStopEpochs=3,
            iterations=75, maxEpochs=2000, kFoldValidationSize=0.20, 
            epochsBetweenPrint=10, verbose=False)

            denormTarget = mlpDataProsessor.denormalizeTarget(targetTestMLP)
            denormResult = mlpDataProsessor.denormalizeTarget(mlp.forward(inputTestMLP))

            denormTarget *= scaleFactor
            denormResult *= scaleFactor
            
            diff = abs(denormTarget - denormResult)
            trainSetAvgDiff[i] = np.mean(diff)
            trainSetStd[i] = np.std(diff)


        print("Statistics for whole set:")
        self.printStatistics(trainSetAvgDiff, trainSetStd)

    def printStatistics(self, trainSetAvgDiff, trainSetStd):

        """
        Prints the given statistics to terminal

        Args:
            trainSetAvgDiff([Double])   : The average diff of each run
            trainSetStd([Double])       : The average standard deviation of 
                                          each run 
        """

        print(trainSetAvgDiff)
        print(trainSetStd)

        bestTrainSetAvgDiff = trainSetAvgDiff.min()
        worstTrainSetAvgDiff = trainSetAvgDiff.max()
        avgTrainSetAvgDiff = np.mean(trainSetAvgDiff)

        bestTrainSetStd = trainSetStd.min()
        worstTrainSetStd = trainSetStd.max()
        avgTrainSetStd = np.mean(trainSetStd)

        trainSetAvgDiffStd = np.std(trainSetAvgDiff)

        print("Train sets average difference range: {:.4f}, {:.4f}".
               format(bestTrainSetAvgDiff,worstTrainSetAvgDiff))

        print("Average train set average difference: {:.4f}".
              format(avgTrainSetAvgDiff))

        print("Train sets standard deviation range: {:.4f}, {:.4f}".
               format(bestTrainSetStd,worstTrainSetStd))

        print("Average train set standard deviation: {:.4f}".
              format(avgTrainSetStd))

        print("Train set average difference standard deviation: {:.4f}".
              format(trainSetAvgDiffStd))



    def applyLogJetColorMap(self, img, logScale=True, maxValue=0):

        """
        Converts image to logaritmic color map

        Args:
            img([[Double]])  : The image 
            logScale(Bool)   : Performs a logaritmic scaling of the colormap
                               if true
            maxValue(float)  : The maximum value for the colormap, will be set
                               to img.max() if value is zero

        """

        if(maxValue == 0):
            maxValue = img.max()
            
        colorMapImg = img/maxValue
        colorMapImg *= 255

        if(logScale):
            colorMapImg[colorMapImg < 1] = 1
            colorMapImg = np.log(colorMapImg)
            colorMapImg = colorMapImg/colorMapImg.max()
            colorMapImg *= 255
    
        colorMapImg = colorMapImg.astype(np.uint8)    
        colorMapImg = cv2.applyColorMap(colorMapImg, cv2.COLORMAP_JET)

        return colorMapImg


    def createColorBarWithLogDepth(self, maxDepth, img, logScale=True, 
                                   scaleFactor=0.001):

        """
        Create a colorbar with text indicating depth of colors. As the image is 
        displayed through a logarithmic transform for better visualisation,
        the colorbar has to compensate for this, thus the relation between 
        depths and colors are exponential

            Args:
                maxDepth(double)             : The max depth in the depth map
                img(MxN matrix, numpy arrays): Image which colorBar should belong to.
                scaleFactor(float)           : Scales the depths by this amount

        """

        colorBar = np.zeros((50, img.shape[1]))

        for i in range(len(colorBar)):
            for j in range(len(colorBar[0])):
                colorBar[i,j] = int(j*255/img.shape[1])

        displayOffset = 20
        coordinates = np.linspace(0, img.shape[1], 9)
        colorspan = np.linspace(0, 256, 9)

        if(logScale):
            colorDepths = (np.exp(colorspan*np.log(255)/255)
                           *maxDepth/255)*scaleFactor
        else:
            colorDepths = colorspan*maxDepth*scaleFactor/255

        colorBar = colorBar.astype(np.uint8)
        colorBar = cv2.applyColorMap(colorBar, cv2.COLORMAP_JET)

        for i in range(len(coordinates)-1):
            
            if(i < 2 or i > len(coordinates)-3):
                color = (255,255,255)
            else:
                color = (0,0,0)

            cv2.circle(colorBar, (int(coordinates[i]),25), 1, color, -1)
            cv2.putText(colorBar, "{:.2f}".format(colorDepths[i]), (int(coordinates[i]),25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


        cv2.circle(colorBar, (int(coordinates[-1]-2),25), 1, color, -1)
        cv2.putText(colorBar, "{:.2f}".format(colorDepths[-1]), (int(coordinates[-1]-displayOffset),25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        separate_white = (np.ones((5, colorBar.shape[1], colorBar.shape[2]))*255).astype(np.uint8)

        imgWithColorBar = np.concatenate((img, separate_white))
        imgWithColorBar = np.concatenate((imgWithColorBar, colorBar))
        imgWithColorBar = np.concatenate((imgWithColorBar, separate_white))

        return imgWithColorBar


if __name__ == '__main__':

    mlpDataInterpreter = MLPDataInterpreter()

    activationFuncs = [tf.nn.relu, tf.nn.relu, tf.nn.sigmoid]
    layers = [15,15]
    usedSegments = [2,3,4,5,6,7,8,9,10,11,12]

    mlpDataInterpreter.runStatistics(layers, activationFuncs, usedSegments, 
                                     imgLeftPath="../Images/CityLeft1.png", 
                                     imgRightPath="../Images/CityRight1.png",
                                     picklePath="../Pickle/City1.p",
                                     MLPIterations=20)


