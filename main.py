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
Main script. See README.md for more informations

Use python 3
"""

import argparse # Command line parsing
import time # Chronometter the timing
import os # For saving the model
from tqdm import tqdm # Progress bar
import tensorflow as tf

from textdata import TextData
from model import Model


def parseArgs():
    """
    Parse the agruments from the given command line
    """
    
    parser = argparse.ArgumentParser()
    
    # Dataset options
    parser.add_argument('--corpus', type=str, default='cornell', help='dataset to choose (cornell)')
    parser.add_argument('--ratioDataset', type=float, default=1.0, help='ratio of dataset used')
    
    # Global options
    parser.add_argument('--seed', type=int, default='123', help='random seed for replication')
    parser.add_argument('--save', type=str, default='save', help='directory to load checkpointed models')
    parser.add_argument('--load', type=str, default='save', help='directory to store checkpointed models')
    
    # Network options
    parser.add_argument('--hiddenSize', type=int, default=300, help='number of hidden units in LSTM')
    
    # Training options
    parser.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
    parser.add_argument('--batchSize', type=int, default=10, help='mini-batch size')
    
    return parser.parse_args()
    
def main():
    """
    Launch the training and/or the interactive mode
    """
    print("Welcome to DeepQA v0.1 !")
    print()
    
    args = parseArgs();
    
    # TODO: Fix seed (WARNING: If dataset shuffling, make sure to do that after saving the 
    # dataset, otherwise, all which cames after the shuffling won't be replicable when 
    # reloading the dataset)
    
    textData = TextData(args)
    #textData.makeLighter(args.ratioDataset) # TODO: Limit size
    model = Model(args, textData)
    
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #saver = tf.train.Saver(tf.all_variables())
        for e in range(args.numEpochs):
            
            print("--- Epoch %d/%d ; (LR=TODO?)" % (e, args.numEpochs))
            print()
            
            batches = textData.getBatches(args.batchSize)
            # TODO: Also update learning parametters eventually
            
            tic = time.clock()
            for nextBatch in tqdm(batches):
                #_, loss = optimizer(feval, params, optimState)
                # optOp.run()
                pass
            toc = time.clock()
            print("Epoch finished in: ", toc-tic)
            
  #local errors = {}
  #local timer = torch.Timer()

  #for i=1, dataset.examplesCount/options.batchSize do
    #collectgarbage()
    
    #local _,tloss = optim.adam(feval, params, optimState)
    #err = tloss[1] -- optim returns a list
  
    #model.decoder:forget()
    #model.encoder:forget()

    #table.insert(errors,err)
    #xlua.progress(i * options.batchSize, dataset.examplesCount)
  #end

  #xlua.progress(dataset.examplesCount, dataset.examplesCount)
  #timer:stop()
  
  #errors = torch.Tensor(errors)
  #print("\n\nFinished in " .. xlua.formatTime(timer:time().real) ..
    #" " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
  #print("\nEpoch stats:")
  #print("  Errors: min= " .. errors:min())
  #print("          max= " .. errors:max())
  #print("       median= " .. errors:median()[1])
  #print("         mean= " .. errors:mean())
  #print("          std= " .. errors:std())
  #print("          ppl= " .. torch.exp(errors:mean()))
            
            
            pass
            #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            #data_loader.reset_batch_pointer()
            #state = model.initial_state.eval()
            #for b in range(data_loader.num_batches):
                #start = time.time()
                #x, y = data_loader.next_batch()
                #feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                #train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                #end = time.time()
                #print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    #.format(e * data_loader.num_batches + b,
                            #args.num_epochs * data_loader.num_batches,
                            #e, train_loss, end - start))
                
                ## Save the model
                #if (e * data_loader.num_batches + b) % args.save_every == 0\
                    #or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    #checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    #saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    #print("model saved to {}".format(checkpoint_path))
                    
    pass

if __name__ == "__main__":
    main()
