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
Train the program by launching it with random parametters
"""

from tqdm import tqdm
import os


def main():
    """
    Launch the training with different parametters
    """

    # TODO: define:
    # step+noize
    # log scale instead of uniform

    # Define parametter: [min, max]
    dictParams = {
        "batchSize": [int, [1, 3]]
        "learningRate": [float, [1, 3]]
        }

    # Training multiple times with different parametters
    for i in range(10):
        # Generate the command line arguments
        trainingArgs = ""
        for keyArg, valueArg in dictParams:
            value = str(random(valueArg[0], max=valueArg[1]))
            trainingArgs += " --" + keyArg + " " + value

        # Launch the program
        os.run("main.py" + trainingArgs)

        # TODO: Save params/results ? or already inside training args ?


if __name__ == "__main__":
    main()
