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
Test the chatbot by launching some unit tests
Warning: it does not check the performances of the program

"""

import unittest
import io
import sys

from chatbot import chatbot


class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.chatbot = chatbot.Chatbot()

    def test_training_simple(self):
        self.chatbot.main([
            '--maxLength', '3', 
            '--numEpoch', '1', 
            '--modelTag', 'unit-test'
        ])

    def test_training_watson(self):
        pass

    def test_testing_all(self):
        pass

    def test_testing_interactive(self):
        progInput = io.StringIO()
        progInput.write('Hi!\n')
        progInput.write('How are you ?\n')
        progInput.write('aersdsd azej qsdfs\n')  # Unknown words
        progInput.write('é"[)=è^$*::!\n')  # Encoding
        progInput.write('ae e qsd, qsd 45 zeo h qfo k zedo. h et qsd qsfjze sfnj zjksdf zehkqf jkzae?\n')  # Too long sentences
        progInput.write('exit\n')

        #sys.stdin = progInput

        #self.chatbot.main(['--test', 'interactive', '--modelTag', 'unit-test'])

    def test_testing_daemon(self):
        pass

if __name__ == '__main__':
    unittest.main()
