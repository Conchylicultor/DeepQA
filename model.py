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

from textdata import TextData
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
        # self.args = args  # Same for the args, could be better than giving args as parameter ??

        # Placeholders
        self.encoderInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget (used for training only)
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.output = None  # Output of the network (without the cost layer fct ?)

        # Construct the graphs
        self.buildNetwork(args)
        self.buildOptimizer(args)

    def buildNetwork(self, args):
        """ Create the computational graph
        Args:
            args: parameters of the model
        """
        forwardOnly = False  # TODO: Define globally

        # Creation of the rnn cell
        with tf.variable_scope("enco_deco_cell") as scope:  # TODO: What does scope really does (just graph visualisation ?) / Use name_scope instead ?
            encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(args.hiddenSize, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
            encoDecoCell = tf.nn.rnn_cell.DropoutWrapper(encoDecoCell)  # TODO: Custom values
            encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * args.numLayers, state_is_tuple=True)

        # TODO: What format use ?? normally int32 (except if word2vec) !!!!
        
        # Network input (placeholders)

        # Batch size * sequence length * input dim (TODO: Variable length sequence !!)
        self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(args.maxLength)]

        self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(args.maxLength)]  # Same sentence length for input and output (Right ?)
        self.decoderTargets = [tf.placeholder(tf.int32,   [None, ]) for _ in range(args.maxLength)]
        self.decoderWeights = [tf.placeholder(tf.float32, [None, ]) for _ in range(args.maxLength)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        decoderOutputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
            embedding_size=25,  # Dimension of each word TODO: args.embeddingSize (or use = args.hiddenSize ?)
            output_projection=None,  # Eventually
            feed_previous=forwardOnly  # When we test (forwardOnly), we use previous output as next input (feed_previous)
        )

        # Finally, we define the loss function
        self.lossFct = tf.nn.seq2seq.sequence_loss(decoderOutputs, self.decoderTargets, self.decoderWeights, self.textData.getVocabularySize())

    def garbageCode(self, args): # TODO: Cleanup garbage code
        # Softmax layer to get the word prediction
        # WARNING: Do not confuse args.sampleSize with self.textData.getSampleSize(), the number of training sample
        assert 0 < args.sampleSize < self.textData.getVocabularySize(), "sampleSize should be smaller than the vocabulary sze"

        with tf.variable_scope("softmax") as scope:
            W = tf.get_variable("proj_w", [args.hiddenSize, self.textData.getVocabularySize()])
            Wt = tf.transpose(W)
            b = tf.get_variable("proj_b", [self.textData.getVocabularySize()])
            #output_projection = (W, b)

            # Define sampled loss function (Only for training)
            def sampledLoss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])  # TODO: What does it do ? Flatten the labels ??
                return tf.nn.sampled_softmax_loss(Wt, b,
                                                  inputs,  # [batch_size, hidden_size] Forward activation
                                                  labels,  # [batch_size, num_true=1] Only one correct label by prediction
                                                  args.sampleSize,  # Number of classes to randomly sample per batch
                                                  self.textData.getVocabularySize())
            #seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
            self.lossFct = sampledLoss

            #self.output = tf.nn.softmax(tf.matmul(output, W) + b)  # Do argmax of the softmax prediction to get the id of the predicted word

        
    def buildOptimizer(self, args):
        """ Initialize the optimizer
        Args:
            args: parametters of the model
        """
        opt = tf.train.AdamOptimizer(
            learning_rate=0.001, 
            beta1=0.9, 
            beta2=0.999, 
            epsilon=1e-08
        )
        self.optOp = opt.minimize(self.lossFct)  #, model.getVariables())
    
    
    def step(self, sess, batch, args):
        """ Forward/training step
        Args:
            sess (tf.Session): a tensorflow session object
            batch (Batch): Input data on testing mode, input and target on output mode
            TODO: (forwardOnly/trainingMode ?) batch
        """
        # TODO: Check with torch lua how the batches are created (encoderInput/Output)

        # Feed the dictionary
        feedDict = {}
        #print('enc_inp:', len(self.encoderInputs))
        #print('batc:', len(batch.inputSeqs))
        for i in range(args.maxLength):
            #print('i:', i)
            #print(self.encoderInputs[i].get_shape())
            #print(len(batch.inputSeqs[i]))
            feedDict[self.encoderInputs[i]]  = batch.inputSeqs[i]
            feedDict[self.decoderInputs[i]]  = batch.targetSeqs[i]
            feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
            feedDict[self.decoderWeights[i]] = batch.weights[i]

        # Run one pass
        sess.run(self.optOp, feedDict)

        # Instead of having session as parameter, try returning just self.optOp and feedDict
        # so that from the main, we can try something like:
        # sess.run(model.step())

        #_, loss = optimizer(feval, params, optimState)
        #_, loss_t, summary = sess.run([self.optOp, self.lossFct, summary_op], feed_dict)

        pass
