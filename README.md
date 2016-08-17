# Deep Q&A

## Presentation

This work try to reproduce the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It use a RNN (seq2seq model) for sentence predictions. It is done using python and tensorflow.

The program is inspired of the [neuralconvo](https://github.com/macournoyer/neuralconvo) from [macournoyer](https://github.com/macournoyer), at least for the loading corpus part.

For now, it use the [Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) corpus but one of the long terms goal is to test it on bigger corpus.

## Installation

The program require the following dependecies (easy to install using pip):
 * python 3
 * tensorflow (tested with v0.9.0)
 * numpy
 * CUDA (for using gpu, see tensorflow [installation page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) for more details
 * nltk (natural language toolkit for tokenized the sentences)
 * tqdm (for the nice progression bars)

The Cornell dataset is already included.

## Running

Multiple trainings (`train.py`) (not complete), single training (`main.py`), production mode (`main.py --test` or `main.py --testInteractive`). Visualisation recorded on `save/summary/` (`tensorboard --logdir save/`). Help and options with `python main.py -h`.
