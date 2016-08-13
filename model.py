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
        self.textData = textData  # Keep a reference on the dataset

        #self.data = data
        #self.target = target
        self.network = None
        self.lossFct = None
        self.optOp = None
        self.output = None  # Output of the network (without the cost layer fct)
        
        # TODO: Use: nb of hidden units args.... ; textData.getVocabularySize() ; gotoken, eostoken

        self.buildNetwork(args)
        self.buildOptimizer(args)
        #self.buildError(args)
        
    def buildNetwork(self, args):
        """ Create the computational graph
        Args:
            args: parameters of the model
        """
        print("Model creation")
        
        # TODO: Create both encoder network and decoder network ??
        
        # Creation of the rnn cell
        with tf.variable_scope("enco_deco_cell") as scope:  # TODO: What does scope really does (just graph visualisation ?) / Use name_scope instead ?
            encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(args.hiddenSize, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
            encoDecoCell = tf.nn.rnn_cell.DropoutWrapper(encoDecoCell)  # TODO: Custom values
            encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * args.numLayers, state_is_tuple=True)
        
        self.network = encoDecoCell
        
        # TODO: What is the input word vector representation (one hot ? word2vec ? just number ?)
        inputDim = 1
        
        # TODO: What format use ?? robably int32 (except if word2vec) !!!!
        
        # Network input (placeholders)
        phData = tf.placeholder(tf.float32, [None, args.maxLength, inputDim])  # Batch size * sequence length * input dim (TODO: Variable length sequence !!)
        #phTarget = tf.placeholder(tf.float32, [None, 21])
        phSeqLength = tf.placeholder(tf.int32, [None])  # Contain the sequence length
        
        # Define the network
        output, state = tf.nn.dynamic_rnn(
            encoDecoCell,
            phData,
            dtype=tf.float32,
            sequence_length=phSeqLength,
        )

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

            self.output = tf.nn.softmax(tf.matmul(output, W) + b)  # Do argmax of the softmax prediction to get the id of the predicted word

        def garbageCode():
            # TODO: Cleanup garbage code
            # Finalize the graph
            forwardOnly = False  # TODO: Modify that later
            if forwardOnly:
                outputs = tf.nn.seq2seq.basic_rnn_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    encoDecoCell,
                )
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                    softmax_loss_function=softmax_loss_function)
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [
                            tf.matmul(output, output_projection[0]) + output_projection[1]
                            for output in self.outputs[b]
                        ]
            else:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x, y: seq2seq_f(x, y, False),
                    softmax_loss_function=softmax_loss_function)

            #tf.nn.seq2seq.basic_rnn_seq2seq

            # Seq2seq model
            #self.forwardSeq2seq = def forwardSeq2seq():
            #    return tf.nn.seq2seq.embedding_attention_seq2seq(
            #       encoder_inputs, decoder_inputs, cell, vocab_size,
            #       vocab_size,hidden_size, output_projection=output_projection,
            #       feed_previous=do_decode)


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
        self.optOp = opt.minimize(self.lossFct) #, model.getVariables())


    def buildError(self, args):
        """ Create the error function
        Args:
            args: parametters of the model
        """
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    
    def step(self, batch):
        """ Forward step
        Args:
            batch: Input data on testing mode, input and target on output mode
            TODO: (forwardOnly/trainingMode ?) batch
        """
        X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
             for _ in range(batch_size)]
        Y = X[:]

        batch.inputSeqs

        # Dimshuffle to seq_len * batch_size
        X = np.array(X).T
        Y = np.array(Y).T

        feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
        feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

        _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)

        pass
