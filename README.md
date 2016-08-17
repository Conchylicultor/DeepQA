# Deep Q&A

## Presentation

This work try to reproduce the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It use a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

The program is inspired of the [neuralconvo](https://github.com/macournoyer/neuralconvo) from [macournoyer](https://github.com/macournoyer), at least for the loading corpus part.

For now, it use the [Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) corpus but one of the long terms goal is to test it on bigger corpus.

## Installation

The program require the following dependecies (easy to install using pip):
 * python 3
 * tensorflow (tested with v0.9.0)
 * numpy
 * CUDA (for using gpu, see TensorFlow [installation page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) for more details
 * nltk (natural language toolkit for tokenized the sentences)
 * tqdm (for the nice progression bars)

The Cornell dataset is already included.

## Running

To train the model, simply run `main.py`. Once trained, you can test the results with `main.py --test` (results generated in 'data/test/samples_predictions.txt') or `main.py --testInteractive` (more fun).

A small script exist (`train.py`) to launch multiple training with different parameters but its not complete yet.

To visualize the computational graph and the cost with [TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html), just run `tensorboard --logdir save/`. For more help and options, use `python main.py -h`.

## Results
