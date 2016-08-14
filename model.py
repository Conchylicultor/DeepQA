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
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf

from textdata import Batch


class Model:
    """
    Implementation of a seq2seq model.
    Achitecture:
        2 LTSM layers
    """
    
    def __init__(self, args, textData):
        """
        Args:
            args: parametters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model

        # Placeholders
        self.encoderInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget (used for training only) (TODO: Could we merge both ?)
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.output = None  # Output of the network (without the cost layer fct ?)

        # Construct the graphs
        self.buildNetwork()
        self.buildOptimizer()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Creation of the rnn cell
        encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
        encoDecoCell = tf.nn.rnn_cell.DropoutWrapper(encoDecoCell)  # TODO: Custom values (WARNING: No dropout when testing !!!)
        encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * self.args.numLayers, state_is_tuple=True)

        # Network input (placeholders)

        # Batch size * sequence length * input dim (TODO: Variable length sequence !!)
        with tf.name_scope('encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLength)]

        with tf.name_scope('decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLength)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLength)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLength)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        decoderOutputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
            embedding_size=self.args.embeddingSize,  # Dimension of each word
            output_projection=None,  # Eventually
            feed_previous=self.args.test  # When we test (self.args.test), we use previous output as next input (feed_previous)
        )

        # Finally, we define the loss function
        self.lossFct = tf.nn.seq2seq.sequence_loss(decoderOutputs, self.decoderTargets, self.decoderWeights, self.textData.getVocabularySize())
        self.attachDetailedSummaries(self.lossFct, 'Loss_fct')  # Keep track of the cost


    def buildOptimizer(self):
        """ Initialize the optimizer
        """
        opt = tf.train.AdamOptimizer(
            learning_rate=self.args.learningRate,
            beta1=0.9, 
            beta2=0.999, 
            epsilon=1e-08
        )
        self.optOp = opt.minimize(self.lossFct)  #, model.getVariables())
    
    
    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
            TODO: (forwardOnly/trainingMode ?) batch
        Return:
            (ops), dict: A tuple of the (training, loss) operators with the associated feed dictionary
        """
        # TODO: Check with torch lua how the batches are created (encoderInput/Output)

        # Feed the dictionary
        feedDict = {}
        for i in range(self.args.maxLength):
            feedDict[self.encoderInputs[i]]  = batch.inputSeqs[i]
            feedDict[self.decoderInputs[i]]  = batch.targetSeqs[i]
            feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
            feedDict[self.decoderWeights[i]] = batch.weights[i]

        # Return one pass operator
        return (self.optOp, self.lossFct), feedDict

    @staticmethod
    def attachDetailedSummaries(var, name):
        """Attach a lot of summaries to a Tensor.
        Args:
            var (tf.tensor): tensor object for which attach the summary
            name (str): name under which the summary will be generated
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
