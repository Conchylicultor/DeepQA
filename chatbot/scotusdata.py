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

import os

"""
Load transcripts from the Supreme Court of the USA.

Available from here:
https://github.com/pender/chatbot-rnn

"""

class ScotusData:
    """
    """

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.lines = self.loadLines(os.path.join(dirName, "scotus"))
        self.conversations = [{"lines": self.lines}]


    def loadLines(self, fileName):
        """
        Args:
            fileName (str): file to load
        Return:
            list<dict<str>>: the extracted fields for each line
        """
        lines = []

        with open(fileName, 'r') as f:
            for line in f:
                l = line[line.index(":")+1:].strip()  # Strip name of speaker.

                lines.append({"text": l})

        return lines


    def getConversations(self):
        return self.conversations
