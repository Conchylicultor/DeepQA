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

from chatbot.textdata import Batch


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
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Creation of the rnn cell
        with tf.variable_scope("chatbot_cell"):  # TODO: How to make this appear on the graph ?
            encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
            #encoDecoCell = tf.nn.rnn_cell.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=1.0)  # TODO: Custom values (WARNING: No dropout when testing !!!)
            encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * self.args.numLayers, state_is_tuple=True)

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]

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
            feed_previous=bool(self.args.test)  # When we test (self.args.test), we use previous output as next input (feed_previous)
        )

        # For testing only
        if self.args.test:
            self.outputs = decoderOutputs
            # TODO: Attach a summary to visualize the output

        # For training only
        else:
            # Finally, we define the loss function
            self.lossFct = tf.nn.seq2seq.sequence_loss(decoderOutputs, self.decoderTargets, self.decoderWeights, self.textData.getVocabularySize())
            tf.scalar_summary('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)
    
    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
