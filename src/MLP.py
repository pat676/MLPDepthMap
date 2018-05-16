import tensorflow as tf
import time
import numpy as np
import random as rand

class MLP():

    """
    A class for createing a basic Multi Layer Perceptron

    Args:
        numInputs(Int)                      : The size of the innput vector 
        numOutputs(Int)                     : The size of the output vector
        numHiddenNodes([Int])               : A vector with the number of hidden 
                                              nodes, the dimension of the vector 
                                              determines the number of hidden layers
        optimizerFuncs                      : The tensor flow optimizers that should 
                                              be used
        (tf.train."optimizer")                When early stop is activated the next 
                                              function in the array is used        
        activationFuncs(tf."activation")    : The array with the activation functions
        lossFunc(tf."lossFunc")             : The function for calculating loss of each 
                                              target
        reduceLossFunc(tf."reduce")         : The reudction func used to reduce the 
                                              lossFunc vector
    """

    #Parameters used by the checkEarlyStopMethode()
    _lastLoss = 1e10
    _numEarlyStopsWithoutImprovement = 0

    def __init__(self, numInputs, numOutputs, numHiddenNodes, optimizerFuncs, 
                 activationFuncs, lossFunc):

        if len(np.shape(numHiddenNodes)) == 0:
            numHiddenNodes = [numHiddenNodes]
        if len(np.shape(optimizerFuncs)) == 0:
            optimizerFuncs = [optimizerFuncs]

        self.numHiddenNodes = numHiddenNodes
        self.numHiddenLayers = np.shape(numHiddenNodes)[0]
        self.numInputs = numInputs
        self.numOutputs = numOutputs

        self.X = tf.placeholder(tf.float32, [None,numInputs])
        self.Y = tf.placeholder(tf.float32, [None,numOutputs])

        self.initWeights()
        self.initBiases()
        self.initResults(activationFuncs)

        self.loss = lossFunc(self.y_estimated, self.Y)
        self.initOptimizerFuncs(optimizerFuncs)

        self.initSession()

    
    def initWeights(self):

        """
        Initializes the weight matricies
        """

        self.W = []
        self.W.append(tf.Variable(tf.random_normal([self.numInputs, self.numHiddenNodes[0]])))
        for i in range(1, self.numHiddenLayers):
            self.W.append(tf.Variable(tf.random_normal([self.numHiddenNodes[i-1], 
                                                        self.numHiddenNodes[i]])))
            
        self.W.append(tf.Variable(tf.random_normal([self.numHiddenNodes[self.numHiddenLayers-1], 
                                                                        self.numOutputs])))
    def initBiases(self):

        """
        Initializes the bias vectors
        """

        self.b = []
        self.b.append(tf.Variable(tf.random_normal([self.numHiddenNodes[0]])))
        for i in range(1, self.numHiddenLayers):
            self.b.append(tf.Variable(tf.random_normal([self.numHiddenNodes[i]])))
        self.b.append(tf.Variable(tf.random_normal([self.numOutputs])))

    def initResults(self, activationFuncs):

        """
        Initializes the result nodes of each layer
        """

        self.results = []
        self.results.append(activationFuncs[0](tf.add(tf.matmul(self.X, self.W[0]),self.b[0])))
        for i in range(1,self.numHiddenLayers):
            self.results.append(activationFuncs[i](tf.add(tf.matmul(self.results[i-1], self.W[i]),self.b[i])))
        self.y_estimated = activationFuncs[-1](tf.add(tf.matmul(self.results[self.numHiddenLayers-1],
                                                                self.W[self.numHiddenLayers]),
                                                                self.b[self.numHiddenLayers]))

    def initOptimizerFuncs(self, optimizerFuncs):

        """
        Initializes the optimizer functions
        """

        self.optimizerFuncs = []
        for func in optimizerFuncs:
            self.optimizerFuncs.append(func.minimize(self.loss))

    def initSession(self):

        """
        Initializes the session
        """

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def train(self, innput, target, batchSize = 10, validSetInputs = None, 
              validSetTargets = None, earlyStopEpochs=2, maxEpochs=100, iterations=100, 
              kFoldValidationSize = 0, epochsBetweenPrint=1, verbose=False):
        
        """
        Trains the mlp.

        Uses the loss, reduction and optimiation functions defined in init to train the MLP

        Args:
            innput ([[Double]])         : The vector with inputs
            target ([[Double]])         : The vector with targets                   
            batchSize(([Double]))       : The size of a training batch
            validSetInputs([[Double]])  : The vector with validation set inputs
            validSetTargets([[Double]]) : The vector with validation set targets
            earlyStopEpochs(Int)        : The number of consectuive epochs without 
                                          improvement before early stop triggers
            maxEpochs(Int)              : The maximum number of epochs 
            iterations(Int)             : The number of iterations in each epoch
            kFoldValidationSize(Int)    : Uses stochastic k-fold validation if this number is 
                                          larger than 0. This number should be between 0 and 1
                                          and describes the amount of the input/targets sets 
                                          that should be used for validation each iteration.  
                                          Setting this number will disregard the valdSets and
                                          batch size.    
            epochsBetweenPrint(Int)     : Progress printing and graphing will be done
                                          each time this amount of epochs is done
            verbose(Bool)               : Determines the amount of prints during training 
            graph(Bool)                 : Displays the progress in a graph if True
            graphHold(Bool)             : Keeps the graph open after training if True
    
        """ 

        currentOptimzer = 0

        #Used for plotting
        losses = []
        time = []

        if kFoldValidationSize >= 1:
            print("ERROR: kFoldValidationSize should be a number between 0 and 1, disregarding k-fold validation!")

        for epoch in range(0, maxEpochs):

            #Create batches
            inputBatch, targetBatch = innput, target
            if kFoldValidationSize > 0 and kFoldValidationSize < 1: #Uses k- fold validation
                inputBatch, targetBatch, validSetInputs, validSetTargets = self.getStochasticKFoldValidSet(innput, target, kFoldValidationSize)
            else:
                inputBatch, targetBatch = self.getBatch(innput, target, batchSize)

            #Print progress if verbose
            if verbose and epoch % epochsBetweenPrint == 0:
                loss = self.session.run(self.loss, feed_dict = {self.X: validSetInputs, self.Y: validSetTargets})
                losses.append(loss)
                time.append(epoch)
                print("Progress: {}%, Current Loss: {:.2e}".format(int((100.0*epoch)/maxEpochs), loss))


            #Train
            for i in range(0, iterations):
                self.trainStep(self.optimizerFuncs[currentOptimzer], inputBatch, targetBatch)
            

            #Do early stop check if spesified
            if earlyStopEpochs > 0 and validSetInputs != None and validSetTargets != None:
                if(self.checkEarlyStop(validSetInputs, validSetTargets, earlyStopEpochs)):
                    currentOptimzer += 1
                    self.trainingCleanup()

                    if(currentOptimzer < len(self.optimizerFuncs)):
                        if verbose:
                            print("Changed to optimizer function: {}".format(currentOptimzer))

                    else:
                        if verbose:
                            print("Early stopped training, no more optimizer functions")
                            loss = self.session.run(self.loss, feed_dict = {self.X: validSetInputs, self.Y: validSetTargets})
                            print("Current Loss: {:.2e}".format(loss))
                        return

        self.trainingCleanup()

    def getBatch(self, innput, target, batchSize):

        """
        Creates a random batch of from innput and target arrays

        Args:
            innput      ([[Double]]): The array of innputs
            targets     ([[Double]]): The corresponding targets array
            batchSize   (Int)       : The size of the batch

        Returns([[Double]], [[Double]]):
            The innputBatch and outputBatch
        """

        #Make sure batch size isn't larger than number of inputs
        if(batchSize > len(innput)):
            batchSize = len(innput) 

        #Choose random indecies
        idx = np.random.choice(np.arange(len(innput)), batchSize, replace=False)

        #Create batches
        inputBatch, targetBatch = [], []
        for i in idx:
            inputBatch.append(innput[i])
            targetBatch.append(target[i])

        return inputBatch, targetBatch

    def checkEarlyStop(self, validSetInputs, validSetTargets, earlyStopEpochs):

        """
        Does a test if the training should perfrom early stop

        Args:
            validSetInputs([[Double]]) : The validation set
            validSetTargets([[Double]]) : The targets for the validation set
            earlyStopIters(Int)         : The number of consectuive epochs without 
                                          improvement before early stop triggers

        Returns (Bool):
            Returns True if early stopp should be performed, else False
        """

        currentLoss = self.session.run(self.loss, feed_dict = {self.X:validSetInputs, self.Y:validSetTargets})
        if currentLoss >= self._lastLoss:
            self._numEarlyStopsWithoutImprovement += 1
            if(self._numEarlyStopsWithoutImprovement >= earlyStopEpochs):
                return True
        else:
            self._numEarlyStopsWithoutImprovement = 0
        self._lastLoss = currentLoss
        return False

    def getStochasticKFoldValidSet(self, innput, target, kFoldValidationSize):

        """
        Extracts a stochastic validation set from the inputs

        Args:
            innputs([[Double]]): The input vecotr 
        """

        validSize = int(round(len(innput) * kFoldValidationSize))
        trainSize = int(len(innput)) - validSize
        inputBatch = innput.copy()
        targetBatch = target.copy()

        idx = np.random.choice(np.arange(len(innput)), validSize, replace=False)

        #Create batches
        validSetInputs, validSetTargets = [], []
        numPops = 0
        for i in idx:
            validSetInputs.append(inputBatch.pop(i - numPops))
            validSetTargets.append(targetBatch.pop(i - numPops))
            numPops += 1

        return inputBatch, targetBatch, validSetInputs, validSetTargets

    def trainStep(self, optimizer, innput, target):

        """
        Performes one training step.

        Args:
            optimizer (tf.train."optimizer"): The optimizer that should be used for training
            innput ([[Double]])             : The input array
            target ([[Double]])             : The target array
        """

        self.session.run(optimizer, feed_dict = {self.X: innput, self.Y:target})

    def trainingCleanup(self):

        """
        Resets private variables used for training
        """

        self.lastLoss = 1e10
        self.numEarlyStopsWithoutImprovement = 0
    
    def forward(self, innput):

        """
        Runs the input through the mlp

        Args:
            innput ([[Double]]): The input array 
        """

        return self.session.run(self.y_estimated, feed_dict = {self.X: innput})



if __name__ == '__main__':
    INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
    OUTPUT_XOR = [[0],[1],[1],[0]]
    
    optimizerFuncs = [tf.train.AdamOptimizer(0.01)]

    lossFunc = tf.losses.mean_squared_error
    activationFuncs = [tf.sigmoid,tf.sigmoid]

    mlp = MLP(2,1,[4], optimizerFuncs, activationFuncs, lossFunc)
    mlp.train(INPUT_XOR, OUTPUT_XOR, validSetInputs = INPUT_XOR, validSetTargets = OUTPUT_XOR, kFoldValidationSize = 0, verbose = True)

    print(mlp.forward(INPUT_XOR))

    