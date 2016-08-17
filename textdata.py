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

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance

from cornelldata import CornellData


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.inputSeqs = []
        self.targetSeqs = []
        self.weights = []
        self.maxInputSeqLen = 0  # Not used
        self.maxTargetSeqLen = 0


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """
    
    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args

        # Path variables
        self.corpusDir = "data/cornell/"
        self.samplesDir = "data/samples/"
        self.samplesName = self._constructName()
        
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary
        
        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]
        
        self.word2id = {}
        self.id2word = {}  # For a rapid conversion
        
        # Limits of the database (perfomances issues)
        # Limit ??? self.maxExampleLen = options.maxExampleLen or 25
        #self.linesLimit = #
        # TODO: Add an option to cut the dataset (less size > faster)
        
        self.loadCorpus(self.samplesDir)
        
        pass

    def _constructName(self):
        """Return the name of the dataset that the program should use with the current parameters.
        Computer from the base name, the given tag (self.args.datasetTag) and the sentence length
        """
        baseName = 'dataset'
        if self.args.datasetTag:
            baseName += '-' + self.args.datasetTag
        return baseName + '-' + str(self.args.maxLength) + '.pkl'

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        if not math.isclose(ratioDataset, 1.0):
            self.shuffle()  # Really ?
            print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        # print("Shuffling the dataset...")
        pass  # TODO

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is defined by the number of samples given.
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """
        # TODO: Modify code to use copies instead of references (keep original database intact)

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for idSample in range(batchSize):
            # Unpack the sample
            sample = samples[idSample]
            inputSeq = sample[0]
            targetSeq = sample[1]

            # Compute max length sequence
            if len(inputSeq) > batch.maxInputSeqLen:
                batch.maxInputSeqLen = len(inputSeq)
            if len(targetSeq) > batch.maxTargetSeqLen:
                batch.maxTargetSeqLen = len(targetSeq)

            # Finalize
            batch.inputSeqs.append(inputSeq)
            batch.targetSeqs.append(targetSeq)

        # Simple hack to truncate the sequence to the right length (TODO: Improve, the sentences too long should be filtered before (probably in the getBatches fct))
        batch.maxInputSeqLen = self.args.maxLength
        batch.maxTargetSeqLen = self.args.maxLength
        for i in range(batchSize):
            assert len(batch.inputSeqs[i]) <= self.args.maxLength
            assert len(batch.targetSeqs[i]) <= self.args.maxLength  # Long sentences should have been filtered during the dataset creation
            if len(batch.inputSeqs[i]) > self.args.maxLength:
                batch.inputSeqs[i] = batch.inputSeqs[i][0:self.args.maxLength]
            if len(batch.targetSeqs[i]) > self.args.maxLength:
                batch.targetSeqs[i] = batch.targetSeqs[i][0:self.args.maxLength]

        # Add padding & define weight
        for i in range(batchSize):  # TODO: Left padding instead of right padding for the input ???
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (batch.maxTargetSeqLen - len(batch.targetSeqs[i])))
            # TODO: Check that we don't modify the originals sequences (=+ vs .append)
            batch.inputSeqs[i]  = batch.inputSeqs[i]  + [self.word2id["<pad>"]] * (batch.maxInputSeqLen  - len(batch.inputSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.word2id["<pad>"]] * (batch.maxTargetSeqLen - len(batch.targetSeqs[i]))

        # Simple hack to reshape the input (TODO: Improve)
        inputSeqsT = []
        targetSeqsT = []
        weightsT = []  # Corrected orientation
        for i in range(self.args.maxLength):
            inputSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                inputSeqT.append(batch.inputSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            inputSeqsT.append(inputSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.inputSeqs = inputSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        return batch

    def getBatches(self):
        """Prepare the batches for the current epoch
        Args:
            args (Obj): parameters were to extract batchSize (int) and maxLength (int)
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()
        
        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            #self.printBatch(batch)  # Debug
            batches.append(batch)
        return batches

    
    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)
    
    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)
        
    def loadCorpus(self, dirName):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(os.path.join(dirName, self.samplesName)):
            datasetExist = True
        
        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')
            # Corpus creation
            cornellData = CornellData(self.corpusDir)
            self.createCorpus(cornellData.getConversations())
            
            # Saving
            print('Saving dataset...')
            self.saveDataset(dirName)  # Saving tf samples
        else:
            print('Loading dataset from {}...'.format(dirName))
            self.loadDataset(dirName)
        
        assert self.padToken == 0
        
        # Plot some stats:
        print('Loaded: {} words, {} QA'.format(len(self.word2id), len(self.trainingSamples)))
        
    def saveDataset(self, dirName):
        """Save samples to file
        Args:
            dirName (str): The directory where to load/save the model
        """
        
        with open(os.path.join(dirName, self.samplesName), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                "word2id": self.word2id,
                "id2word": self.id2word,
                "trainingSamples": self.trainingSamples
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, dirName):
        """Load samples from file
        Args:
            dirName (str): The directory where to load the model
        """
        with open(os.path.join(dirName, self.samplesName), 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.trainingSamples = data["trainingSamples"]
            
            self.padToken = self.word2id["<pad>"]
            self.goToken = self.word2id["<go>"]
            self.eosToken = self.word2id ["<eos>"]
            self.unknownToken = self.word2id["<unknown>"] # Restore special words


    def createCorpus(self, conversations):
        """Extract all data from the given vocabulary
        """
        # Add standard tokens
        self.padToken = self.getWordId("<pad>")  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId("<go>")  # Start of sequence
        self.eosToken = self.getWordId("<eos>")  # End of sequence
        self.unknownToken = self.getWordId("<unknown>")  # Word dropped from vocabulary
        
        # Preprocessing data

        for conversation in tqdm(conversations, desc="Extract conversations"):
            self.extractConversation(conversation)

        # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, conversation):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a convesation object containing the lines to extract
        """
            
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine  = conversation["lines"][i]
            targetLine = conversation["lines"][i+1]
            
            inputWords  = self.extractText(inputLine ["text"])
            targetWords = self.extractText(targetLine["text"], True)
            
            if not inputWords or not targetWords:  # Filter wrong samples (if one of the list is empty)
                #tqdm.write("Error with some sentences. Sample ignored.")
                pass
            else:
                #print('---------------')
                #self.playASequence(inputWords)
                #self.playASequence(targetWords)

                inputWords.reverse()  # Reverse inputs (and not outputs), little tricks as defined on the original seq2seq paper
                
                targetWords.insert(0, self.goToken)
                targetWords.append(self.eosToken)  # Add the end of string

                self.trainingSamples.append([inputWords, targetWords])


    def extractText(self, line, isTarget=False):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
            isTarget (bool): Define the question on the answer
        Return:
            list<int>: the list of the word ids of the sentence
        """
        words = []

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if not isTarget:
                i = len(sentencesToken)-1 - i

            tokens = nltk.word_tokenize(sentencesToken[i])

            # If the total length is not too big, we still can add one more sentence
            if len(words) + len(tokens) <= self.args.maxLength - 2*int(isTarget):  # For the target, we need to taken into account <go> and <eos>
                tempWords = []
                for token in tokens:
                    tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

                if isTarget:
                    words = words + tempWords
                else:
                    words = tempWords + words
            else:
                break  # We reach the max length already

        return words

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # TODO: Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # Get the id if the word already exist
        id = self.word2id.get(word, -1)
        
        # If not, we create a new entry
        if id == -1:
            if create:
                id = len(self.word2id)
                self.word2id[word] = id
                self.id2word[id] = word
            else:
                id = self.unknownToken
        
        return id

    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        print('Input (should be inverted, as on the paper):')
        for seq in batch.inputSeqs:
            self.playASequence(seq)
        print('Target:')
        for seq in batch.targetSeqs:
            self.playASequence(seq)

    def playASequence(self, sequence):
        """Print the words associated to a sequence
        Args:
            sequence (list<int>): the sentence to print
        """
        if not sequence:
            return

        for w in sequence[:-1]:
            print(self.id2word[w], end=' - ')
        print(self.id2word[sequence[-1]])  # endl

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """
        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences
        #self.playASequence(wordIds)  # TODO: Not printed here but external (as for deco2sentence return 2 values)

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        words = []

        def argmax(lst):
            return max(enumerate(lst), key=lambda x: x[1])[0]  # TODO: Check validity

        # Choose the words with the highest prediction score
        lengthSentence = 0
        for index, out in enumerate(decoderOutputs):  # For each predicted words
            wordId = np.argmax(out)
            words.append(self.id2word[wordId])
            if lengthSentence == 0 and (wordId == self.eosToken or index == len(decoderOutputs)-1):  # End of generated sentence
                lengthSentence = index

        return ' '.join(words[1:lengthSentence]), ' - '.join(words)  # Some cleanup: We remove the go token and everything after eos

    def playADialog(self):
        """Print a random dialogue from the dataset
        """
        pass
