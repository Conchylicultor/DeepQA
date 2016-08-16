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
        self.args = None

        self.textData = None  # Dataset
        self.model = None  # Sequence to sequence model

        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None

    @staticmethod
    def parseArgs():
        """
        Parse the arguments from the given command line
        """

        parser = argparse.ArgumentParser()

        # Global options
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--test', action='store_true', help='if present, launch the program try to answer all sentences from data/test/')  # TODO: Not present yet
        globalArgs.add_argument('--testInteractive', action='store_true', help='if present, launch the interactive testing mode where the user can wrote his own sentences')  # TODO: Not present yet
        globalArgs.add_argument('--createDataset', action='store_true', help='if present, the program will only generate the dataset from the corpus (no training/testing)')
        globalArgs.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the modelDir directory (Warning: the model will be destroyed)')
        globalArgs.add_argument('--modelDir', type=str, default='save/model', help='directory to store/load checkpoints of the models')
        globalArgs.add_argument('--device', type=str, default=None, help='\'cpu\' or \'gpu\' (Warning: make sure you have enough free RAM, like +4GB), allow to choose which hardware use')  # TODO
        globalArgs.add_argument('--seed', type=int, default=None, help='random seed for replication')

        # Dataset options
        datasetArgs = parser.add_argument_group('Dataset options')
        datasetArgs.add_argument('--corpus', type=str, default='cornell', help='dataset to choose (Cornell)')  # Only one corpus right now
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

        # Saver/summaries  # TODO: Synchronize writer, saver and globStep (saving/loading from the same place) (with subfolder name created from args ??)
        summariesRootDir = 'save/summaries/'
        idRun = 0
        while os.path.exists(summariesRootDir + str(idRun)):
            idRun += 1
        self.writer = tf.train.SummaryWriter(summariesRootDir + str(idRun))  # Define a custom name (created from the args) ?
        self.saver = tf.train.Saver()

        # TODO: Fixed seed (WARNING: If dataset shuffling, make sure to do that after saving the
        # dataset, otherwise, all which cames after the shuffling won't be replicable when
        # reloading the dataset)

        # Running session

        with tf.Session() as sess:
            print('Initialize variables...')
            tf.initialize_all_variables().run()
            self.writer.add_graph(sess.graph)

            self.managePreviousModel(sess)  # Reload the model (eventually)

            if self.args.test:
                self.mainTest(sess)  # TODO: test and testInteractive
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
        globStep = 0

        # TODO: If restoring a model, also restoring globStep ? progression bar ? current batch ? continue summarize at same point

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
                self.writer.add_summary(summary, globStep)
                globStep += 1

                # Checkpoint
                if globStep % self.args.saveEvery == 0:
                    tqdm.write('Checkpoint reached: saving model...')
                    self.saver.save(sess, self.args.modelDest)  # Warning: self.args.modelDest is defined in managePreviousModel
                    tqdm.write('Model saved.')
            toc = time.perf_counter()

            print("Epoch finished in: {}s".format(toc-tic))  # TODO: Better time format

    def mainTest(self, sess):
        """ Try predicting the sentences
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
        MODEL_NAME = 'model.ckpt'

        print('WARNING: ', end='')
        self.args.modelDest = os.path.join(self.args.modelDir, MODEL_NAME)  # Warning: Creation of new variable (accessible from train() for saving the model)
        if os.path.isfile(self.args.modelDest):
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.args.modelDest))
                os.remove(self.args.modelDest)
                os.remove(self.args.modelDest + '.meta')
            else:
                print('Restoring previous model from {}'.format(self.args.modelDest))
                self.saver.restore(sess, self.args.modelDest)
                print('Model restored.')
        else:
            print('No previous model found, starting from clean directory: {}'.format(self.args.modelDir))
            os.makedirs(self.args.modelDir)

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
