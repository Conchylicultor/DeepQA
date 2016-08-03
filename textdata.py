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
Loads the dialogue corpus, builds the vocabulary
"""

#import tensorflow as tf
import nltk # For tokenize
from tqdm import tqdm # Progress bar
import pickle # Saving the data
import os # Checking file existance

from cornelldata import CornellData

class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """
    
    def __init__(self, args):
        """Load all conversations
        Args:
            args: parametters of the model
        """
        # Path variables
        self.corpusDir = "data/cornell/"
        self.samplesDir = "data/samples/"
        self.samplesName = "dataset.pkl"
        
        self.goToken = -1 # Start of sequence
        self.eosToken = -1 # End of sequence
        self.unknownToken = -1 # Word dropped from vocabulary
        
        self.trainingSamples = [] # 2d array containing each question and his answer
        
        self.word2id = {}
        self.id2word = {} # For a rapid conversion
        
        # Limits of the database (perfomances issues)
        # Limit ??? self.maxExampleLen = options.maxExampleLen or 25
        #self.linesLimit = #
        
        self.loadCorpus(self.samplesDir)
        
        pass
    
    def loadCorpus(self, dirName):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(dirName+self.samplesName):
            datasetExist = True
        
        if not datasetExist: # Fist time we load the database: creating all files
            print('Training samples not found. Creating dataset...')
            # Corpus creation
            cornellData = CornellData(self.corpusDir)
            self.createCorpus(cornellData.getConversations())
            
            # Saving
            print('Saving dataset...')
            self.saveDataset(dirName) # Saving tf samples
        else:
            print('Loading dataset from %s...' % (dirName))
            self.loadDataset(dirName)
            
            pass # TODO
        
        # TODO: Shuffle the dataset
        
        # Plot some stats:
        print('Loaded: %d words, %d QA' % (len(self.word2id), len(self.trainingSamples)))
        
    def saveDataset(self, dirName):
        """Save samples to file
        Args:
            dirName (str): The directory where to load/save the model
        """
        
        with open(dirName + self.samplesName, 'wb') as handle:
            data = {
                "word2id": self.word2id,
                "id2word": self.id2word,
                "trainingSamples": self.trainingSamples
                }
            pickle.dump(data, handle, -1) # Using the highest protocol available

    def loadDataset(self, dirName):
        """Load samples from file
        Args:
            dirName (str): The directory where to load the model
        """
        with open(dirName + self.samplesName, 'rb') as handle:
            data = pickle.load(handle)
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.trainingSamples = data["trainingSamples"]
            
            self.goToken = self.word2id["<go>"]
            self.eosToken = self.word2id ["<eos>"]
            self.unknownToken = self.word2id["<unknown>"] # Restore special words
            
            
    
    def createCorpus(self, conversations):
        """Extract all data from the given vocabulary
        """
        # Add standard tokens
        self.goToken = self.makeWordId("<go>") # Start of sequence
        self.eosToken = self.makeWordId("<eos>") # End of sequence
        self.unknownToken = self.makeWordId("<unknown>") # Word dropped from vocabulary
        
        # Prepocessing data

        for conversation in tqdm(conversations, desc="Extract conversations"):
            self.extractConversation(conversation)
        
        # TODO: Shuffling (before saving ?)
        
        # TODO: clear trainingSample after saving ?
    
    def extractConversation(self, conversation):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a convesation object containing the lines to extract
        """
            
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1): # We ignore the last line (no answer for it)
            inputLine  = conversation["lines"][i]
            targetLine = conversation["lines"][i+1]
            
            inputWords  = self.extractText(inputLine ["text"])
            targetWords = self.extractText(targetLine["text"], True)
            
            #print(inputLine)
            #print(targetLine)
            #print(inputWords)
            #print(targetWords)
            
            if not inputWords or not targetWords: # If one of the list is empty
                tqdm.write("Error with some sentences. Sample ignored.")
                if inputWords:
                    tqdm.write(inputLine["text"])
                if targetWords:
                    tqdm.write(targetLine["text"])
            else:
                inputWords.reverse() # Reverse inputs (apparently not the output. Why ?)
                
                targetWords.insert(0, self.goToken)
                targetWords.append(self.eosToken) # Add the end of string

                self.trainingSamples.append([inputWords, targetWords])
                #self.trainingSamples.append([tf.constant(inputWords), tf.constant(targetWords)]) # tf.cst ? or keep as array (to feed placeholder) ?
        
    
    def extractText(self, line, isTarget=False):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
            isTarget (bool): Define the question on the answer
        Return:
            list<int>: the list of the word ids of the sentence
        """
        words = []
        
        # TODO !!!
        # If answer: we only keep the last sentence
        # If question: we only keep the first sentence
        
        tokens = nltk.word_tokenize(line)
        for token in tokens: # TODO: Limit size (if sentence too long) ?
            words.append(self.makeWordId(token)) # Create the vocabulary and the training sentences
        
        return words

    def makeWordId(self, word):
        """Add a word to the dictionary
        Args:
            word (str): word to add
        Return:
            int: the id of the word created
        """
        
        # Get the id if the word already exist
        id = self.word2id.get(word, -1)
        
        # If not, we create a new entry
        if id == -1:
            id = len(self.word2id)
            self.word2id[word] = id
            self.id2word[id] = word
        
        return id

    def playADialog():
        """Print a random dialogue from the dataset
        """
        pass
