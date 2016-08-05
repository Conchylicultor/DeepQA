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
        #self.data = data
        #self.target = target
        self.network = None
        self.optimizer = None
        self.error = None
        
        # TODO: Use: nb of hidden units args.... ; textData.getVocabularySize() ; gotoken, eostoken
        
        self.buildNetwork(args)
        #self.buildOptimizer(args)
        #self.buildError(args)
        
    def buildNetwork(self, args):
        """ Create the computational graph
        Args:
            args: parametters of the model
        """
        print("Model creation")
        
        lstm = tf.nn.rnn_cell.BasicLSTMCell(args.hiddenSize, state_is_tuple=True)
        
        network = tf.nn.rnn_cell.GRUCell(args.hiddenSize)  # Or LSTMCell(args.hiddenSize)
        #network = tf.nn.rnn_cell.DropoutWrapper(network, output_keep_prob=dropout)
        #network = tf.nn.rnn_cell.MultiRNNCell([network] * num_layers)
        
        max_length = 100

        #input = tf.placeholder(tf.float32, [None, max_length, 28]) # Batch size x time steps x data width (here 1 ?)
        #output, state = tf.nn.rnn.dynamic_rnn(network, input, dtype=tf.float32)
        
        ## Initial state of the LSTM memory.
        #state = tf.zeros([batch_size, lstm.state_size])

        #loss = 0.0
        #for current_batch_of_words in words_in_dataset:
            ## The value of state is updated after processing each batch of words.
            #output, state = lstm(current_batch_of_words, state)

            ## The LSTM output can be used to make next word predictions
            #logits = tf.matmul(output, softmax_w) + softmax_b
            #probabilities = tf.nn.softmax(logits)
            #loss += loss_function(probabilities, target_words)
            
        #tf.nn.rnn_cell.BasicLSTMCell.__init__(num_units, forget_bias=1.0, input_size=None, state_is_tuple=False, activation=tanh)
        

        
    def buildOptimizer(self, args):
        """ Initialize the optimizer
        Args:
            args: parametters of the model
        """
        opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        cost
        # TODO: optOp = opt.minimize(cost, model.getVariables())
    
    def buildError(self, args):
        """ Create the error function
        Args:
            args: parametters of the model
        """
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    def computeLength(self, data):
        """ Compute the sequence length
        Args:
            data: the input sequence
        """
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
    