# Deep Q&A

This work try to reproduce the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It use a RNN (seq2seq model) for sentence predictions.

The program is inspared of the [neuralconvo](https://github.com/macournoyer/neuralconvo) from [macournoyer](https://github.com/macournoyer) and is basically the tensorflow version of it.

For now, it use the [Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) corpus but one of the long terms goal is to test it on bigger corpus.

Dependencies:
 * python 3
 * tensorflow
 * nlpt
 * tqdm
