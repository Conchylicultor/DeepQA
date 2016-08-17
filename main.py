#!/usr/bin/env python3

# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Main script. See README.md for more information

Use python 3
"""

import argparse  # Command line parsing
import configparser  # Saving the models parameters
import time  # Chronometer
import os  # Files management
from tqdm import tqdm  # Progress bar
import tensorflow as tf

from textdata import TextData
from model import Model


class Main:
    """
    Main class which launch the training or testing mode
    """

    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = None

        # Task specific object
        self.textData = None  # Dataset
        self.model = None  # Sequence to sequence model

        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.modelDir = ''  # Where the model is saved
        self.globStep = 0  # Represent the number of iteration for the current model

        # Filename and directories constants
        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.TEST_DIR = 'data/test'
        self.TEST_IN_NAME = 'samples.txt'
        self.TEST_OUT_NAME = 'samples_predictions.txt'

    @staticmethod
    def parseArgs():
        """
        Parse the arguments from the given command line
        """

        parser = argparse.ArgumentParser()

        # Global options
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--test', action='store_true', help='if present, launch the program try to answer all sentences from data/test/')
        globalArgs.add_argument('--testInteractive', action='store_true', help='if present, launch the interactive testing mode where the user can wrote his own sentences')
        globalArgs.add_argument('--createDataset', action='store_true', help='if present, the program will only generate the dataset from the corpus (no training/testing)')
        globalArgs.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
        globalArgs.add_argument('--keepAll', action='store_true', help='If this option is set, all saved model will be keep (Warning: make sure you have enough free disk space or increase saveEvery)')
        globalArgs.add_argument('--modelTag', type=str, default=None, help='tag to differentiate which model to store/load')
        globalArgs.add_argument('--device', type=str, default=None, help='\'gpu\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')
        globalArgs.add_argument('--seed', type=int, default=None, help='random seed for replication')

        # Dataset options
        datasetArgs = parser.add_argument_group('Dataset options')
        datasetArgs.add_argument('--corpus', type=str, default='cornell', help='corpus on which extract the dataset. Only one corpus available right now (Cornell)')
        datasetArgs.add_argument('--datasetTag', type=str, default=None, help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'
        datasetArgs.add_argument('--ratioDataset', type=float, default=1.0, help='ratio of dataset used to avoid using the whole dataset')  # Not implemented, useless ?
        datasetArgs.add_argument('--maxLength', type=int, default=10, help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')

        # Network options
        nnArgs = parser.add_argument_group('Network options', 'architecture related option')
        nnArgs.add_argument('--hiddenSize', type=int, default=256, help='number of hidden units in each RNN cell')
        nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
        nnArgs.add_argument('--embeddingSize', type=int, default=25, help='embedding size of the word representation')

        # Training options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
        trainingArgs.add_argument('--saveEvery', type=int, default=1000, help='nb of mini-batch step before creating a model checkpoint')
        trainingArgs.add_argument('--batchSize', type=int, default=10, help='mini-batch size')
        trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='Learning rate')

        return parser.parse_args()


    def main(self):
        """
        Launch the training and/or the interactive mode
        """
        print('Welcome to DeepQA v0.1 !')
        print()
        print('Tensorflow detected: v{}'.format(tf.__version__))

        # General initialisation

        self.args = self.parseArgs()

        #tf.logging.set_verbosity(tf.logging.INFO) # DEBUG, INFO, WARN (default), ERROR, or FATAL

        if self.args.testInteractive:  # Training or testing mode
            self.args.test = True

        self.loadModelParams()  # Update the self.modelDir and self.globStep, for now, not used when loading Model (but need to be called before _getSummaryName)

        self.textData = TextData(self.args)
        # TODO: Add a debug mode which would randomly play some sentences
        # TODO: For now, the model are trained for a specific dataset (because of the maxLength which define the
        # vocabulary). Add a compatibility mode which allow to launch a model trained on a different vocabulary (
        # remap the word2id/id2word variables).
        if self.args.createDataset:
            print('Dataset created! Thanks for using our program')
            return  # No need to go futher

        with tf.device(self.getDevice()):
            self.model = Model(self.args, self.textData)

        # Saver/summaries
        self.writer = tf.train.SummaryWriter(self._getSummaryName())
        self.saver = tf.train.Saver()

        # TODO: Fixed seed (WARNING: If dataset shuffling, make sure to do that after saving the
        # dataset, otherwise, all which cames after the shuffling won't be replicable when
        # reloading the dataset)

        # Running session

        with tf.Session() as sess:
            print('Initialize variables...')
            tf.initialize_all_variables().run()

            self.managePreviousModel(sess)  # Reload the model (eventually)

            if self.args.test:
                if self.args.testInteractive:
                    self.mainTestInteractive(sess)
                else:
                    print('Start predicting...')
                    self.predictTestset(sess)
                    print('Prediction done')
            else:
                self.mainTrain(sess)

        print("The End! Thanks for using our program")

    def mainTrain(self, sess):
        """ Training loop
        Args:
            sess: The current running session
        """

        # Specific training dependent loading

        self.textData.makeLighter(self.args.ratioDataset)  # Limit the number of training samples

        mergedSummaries = tf.merge_all_summaries()  # Define the summary operator (Warning: Won't appear on the tensorboard graph)
        if self.globStep == 0:  # Not restoring from previous run
            self.writer.add_graph(sess.graph)  # First time only

        # TODO: If restoring a model, also progression bar ? current batch ?

        print('Start training...')

        for e in range(self.args.numEpochs):

            print("--- Epoch {}/{} ; (lr={})".format(e, self.args.numEpochs, self.args.learningRate))
            print()

            batches = self.textData.getBatches()  # TODO: Shuffle
            # TODO: Also update learning parameters eventually

            tic = time.perf_counter()
            for nextBatch in tqdm(batches, desc="Training"):
                # Training pass
                ops, feedDict = self.model.step(nextBatch)
                assert len(ops) == 2  # training, loss
                _, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)  # TODO: Get the returned loss, return the prediction (testing mode only)
                self.writer.add_summary(summary, self.globStep)
                self.globStep += 1

                # Checkpoint
                if self.globStep % self.args.saveEvery == 0:
                    tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
                    self.saveModelParams()
                    self.saver.save(sess, self._getModelName())
                    tqdm.write('Model saved.')
            toc = time.perf_counter()

            print("Epoch finished in: {}s".format(toc-tic))  # TODO: Better time format

    def predictTestset(self, sess, saveName=None):
        """ Try predicting the sentences from the samples.txt file
        Args:
            sess: The current running session
            saveName: Name where to save the predictions
        """
        if not saveName:
            saveName = os.path.join(self.TEST_DIR, self.TEST_OUT_NAME)

        with open(os.path.join(self.TEST_DIR, self.TEST_OUT_NAME), 'r') as f:
            lines = f.readlines()

        with open(saveName, 'w') as f:
            for line in lines:
                question = line

                batch = self.textData.sentence2enco(question)
                if not batch:
                    continue  # Back to the beginning, try again
                ops, feedDict = self.model.step(batch)
                output = sess.run(ops[0], feedDict)
                answer, answerComplete = self.textData.deco2sentence(output)

                f.write('Q: {}'.format(question))  # The endl is already included on the question string
                f.write('A: {}\n'.format(answerComplete))
                f.write('\n')

    def mainTestInteractive(self, sess):
        """ Try predicting the sentences that the user will enter in the console
        Args:
            sess: The current running session
        """
        print('Testing: Launch interactive mode:')
        print('')
        print('Welcome to the interactive mode, here you can ask to Deep Q&A the sentence you want. Don\'t have high '
              'expectation. Type \'exit\' or just press ENTER to quit the program. Have fun.')
        question = None
        while question != '' and question != ':q' and question != 'exit':
            question = input('Q:')

            batch = self.textData.sentence2enco(question)
            self.textData.playASequence(batch.inputSeqs[0])
            if not batch:
                continue  # Back to the beginning, try again
            ops, feedDict = self.model.step(batch)
            output = sess.run(ops[0], feedDict)
            answer, answerComplete = self.textData.deco2sentence(output)

            print('A:', answer)
            print(answerComplete)

    def managePreviousModel(self, sess):
        """ Restore or reset the model
        Args:
            sess: The current running session
        """

        # TODO: Check all possible cases!

        print('WARNING: ', end='')

        modelName = self._getModelName()

        if os.listdir(self.modelDir):  # Warning: This will not work if we changed the keepAll option
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.modelDir))  # Ask for confirmation ?

                filelist = [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir)]
                for f in filelist:
                    print('Removing {}'.format(f))
                    os.remove(f)
            else:
                print('Restoring previous model from {}'.format(modelName))
                self.saver.restore(sess, modelName)  # Will crash when --reset is not activated and the model has not been saved yet
                print('Model restored.')
        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))
            if not os.path.exists(self.modelDir):  # The directory can still contain previous models (keepAll changed) or could be empty (previous model halted before saving)
                os.makedirs(self.modelDir)  # Make sure the directory exist (otherwise will crash when saving), useless because the summary object does this for us
            else:
                assert not os.listdir(self.modelDir)  # The directory should be empty (keepAll changed not supported yet) (TODO: Will always rise because of summary)

    def loadModelParams(self):
        """ Load the some values associated with the current model, like the current globStep value
        For now, this function does not need to be called before loading the model (no parameters restored). However,
        the modelDir name will be initialized here so it is required to call this function before managePreviousModel(),
        _getModelName() or _getSummaryName()
        Warning: if you modify this function, make sure the changes mirror saveModelParams
        """
        # Compute the current model path
        self.modelDir = self.MODEL_DIR_BASE
        if self.args.modelTag:
            self.modelDir += '-' + self.args.modelTag

        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if os.path.exists(configName):
            config = configparser.ConfigParser()
            config.read(configName)
            self.globStep = config['General'].getint('globStep')

    def saveModelParams(self):
        """ Save the params of the model, like the current globStep value
        Warning: if you modify this function, make sure the changes mirror loadModelParams
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['globStep'] = str(self.globStep)
        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

        # TODO: Save all parameters (also datasetName ?) to keep a track of those

    def _getSummaryName(self):
        """ Parse the argument to decide were to save the summary, at the same place that the model
        The folder could already contain logs if we restore the training, those will be merged
        Return:
            str: The path and name of the summary
        """
        return self.modelDir

    def _getModelName(self):
        """ Parse the argument to decide were to save/load the model
        This function is called at each checkpoint and the first time the model is load. If keepAll option is set, the
        globStep value will be included in the name.
        Return:
            str: The path and name were the model need to be saved
        """
        modelName = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        if self.args.keepAll:  # We do not erase the previously saved model by including the current step on the name
            modelName += '-' + str(self.globStep)
        return modelName

    def getDevice(self):
        """ Parse the argument to decide on which device run the model
        Return:
            str: The name of the device on which run the program
        """
        if self.args.device == 'cpu':
            return '"/cpu:0'
        elif self.args.device == 'gpu':
            return '/gpu:0'
        elif self.args.device is None:  # No specified device (default)
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None


if __name__ == "__main__":
    program = Main()
    program.main()
