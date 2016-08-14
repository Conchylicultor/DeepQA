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


def parseArgs():
    """
    Parse the arguments from the given command line
    """
    
    parser = argparse.ArgumentParser()

    # Global options
    globalArgs = parser.add_argument_group('Global options')
    globalArgs.add_argument('--test', action='store_true', help='if present, launch the program try to answer all sentences from data/test/')  # TODO: Not present yet
    globalArgs.add_argument('--testInteractive', action='store_true', help='if present, launch the interactive testing mode where the user can wrote his own sentences')  # TODO: Not present yet
    globalArgs.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the modelDir directory (Warning: the model will be destroyed)')
    globalArgs.add_argument('--modelDir', type=str, default='save/model', help='directory to store/load checkpoints of the models')
    globalArgs.add_argument('--seed', type=int, default=None, help='random seed for replication')

    # Dataset options
    datasetArgs = parser.add_argument_group('Dataset options')
    datasetArgs.add_argument('--corpus', type=str, default='cornell', help='dataset to choose (Cornell)')
    datasetArgs.add_argument('--ratioDataset', type=float, default=1.0, help='ratio of dataset used')
    datasetArgs.add_argument('--maxLength', type=int, default=50, help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')

    # Network options
    nnArgs = parser.add_argument_group('Network options', 'architecture related option')
    nnArgs.add_argument('--hiddenSize', type=int, default=256, help='number of hidden units in each RNN cell')
    nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
    nnArgs.add_argument('--embeddingSize', type=int, default=25, help='embedding size of the word representation')
    
    # Training options
    trainingArgs = parser.add_argument_group('Training options')
    trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
    trainingArgs.add_argument('--saveEvery', type=int, default=300, help='nb of mini-batch step before creating a model checkpoint')  # TODO: Tune
    trainingArgs.add_argument('--batchSize', type=int, default=10, help='mini-batch size')
    trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='Learning rate')

    return parser.parse_args()


def main():
    """
    Launch the training and/or the interactive mode

    TODO: Could create a class, it would allow to divide the code in more functions. As init members, the main class
    would have object like model, writer, saver
    As fcts, it would have managePreviousModel() which would restore or reset models, and of course main()
    It would be useful for cleaner code to separate testing/training
    """
    print('Welcome to DeepQA v0.1 !')
    print()
    print('Tensorflow detected: v%s' % tf.__version__);

    args = parseArgs()

    MODEL_NAME = 'model.ckpt'

    #tf.logging.set_verbosity(tf.logging.INFO) # DEBUG, INFO, WARN (default), ERROR, or FATAL

    # TODO: Fixed seed (WARNING: If dataset shuffling, make sure to do that after saving the
    # dataset, otherwise, all which cames after the shuffling won't be replicable when 
    # reloading the dataset)

    textData = TextData(args)
    textData.makeLighter(args.ratioDataset)  # Limit the number of training samples
    model = Model(args, textData)

    # Saver/summaries  # TODO: Synchronize writer, saver and globStep (saving/loading from the same place) (with subfolder name created from args ??)
    mergedSummaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("save/summary")  # Define a custom name (created from the args) ?
    saver = tf.train.Saver()
    globStep = 0
    
    with tf.Session() as sess:
        print('Initialize variables...')
        tf.initialize_all_variables().run()
        writer.add_graph(sess.graph)

        print('Initialisation done. Managing previous model...')

        print('WARNING: ', end='')
        modelDest = os.path.join(args.modelDir, MODEL_NAME)
        if os.path.isfile(modelDest):
            if args.reset:
                print('Reset: Destroying previous model at %s' % modelDest)
                os.remove(modelDest)
            else:
                print('Restoring previous model from %s' % modelDest)
                saver.restore(sess, modelDest)
                print("Model restored. ", end='')
        else:
            print('No previous model found, starting from clean directory: %s' % args.modelDir)
            os.makedirs(args.modelDir)

        print('Start training...')

        for e in range(args.numEpochs):

            print("--- Epoch %d/%d ; (lr=%f)" % (e, args.numEpochs, args.learningRate))
            print()

            batches = textData.getBatches(args)  # TODO: Shuffle
            # TODO: Also update learning parameters eventually

            tic = time.clock()  # TODO: or time.time()
            for nextBatch in tqdm(batches, desc="Training"):
                # Training pass
                ops, feedDict = model.step(nextBatch)
                assert len(ops) == 2  # training, loss
                _, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)  # TODO: Get the returned loss, return the prediction (testing mode only)
                writer.add_summary(summary, globStep)
                globStep += 1

                # Checkpoint
                if globStep % args.saveEvery == 0:
                    tqdm.write('Checkpoint reached: saving model...', end='')
                    saver.save(sess, modelDest)
                    tqdm.write('Model saved.')
            toc = time.clock()

            print("Epoch finished in: %2fs" % (toc-tic))
    
    print("The End! Thanks for using our program")

if __name__ == "__main__":
    main()
