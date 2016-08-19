# Deep Q&A

## Presentation

This work try to reproduce the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It use a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

The program is inspired of the Torch [neuralconvo](https://github.com/macournoyer/neuralconvo) from [macournoyer](https://github.com/macournoyer), at least for the loading corpus part.

For now, it use the [Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) corpus but one of the long terms goal is to test it on bigger corpus.

## Installation

The program require the following dependecies (easy to install using pip):
 * python 3
 * tensorflow (tested with v0.9.0)
 * numpy
 * CUDA (for using gpu, see TensorFlow [installation page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) for more details)
 * nltk (natural language toolkit for tokenized the sentences)
 * tqdm (for the nice progression bars)

The Cornell dataset is already included.

## Running

To train the model, simply run `main.py`. Once trained, you can test the results with `main.py --test` (results generated in 'save/model/samples_predictions.txt') or `main.py --testInteractive` (more fun).

A small script exist (`train.py`) to launch multiple training with different parameters but its not complete yet.

To visualize the computational graph and the cost with [TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html), just run `tensorboard --logdir save/`. For more help and options, use `python main.py -h`.

## Results

Here are some preliminary results (uncensored). For performance issue, I limited the sentence length to 5. The network seems to learn correctly, but she's not really chatty:

    Q: Hi
    A: Hi.
    
    Q: What is your name ?
    A: Laura.
    
    Q: What does that mean ?
    A: I dunno.
    
    Q: How old are you ?
    A: thirty-five.
    
    Q: Are you alive ?
    A: No.
    
    Q: Will Google hire me ?
    A: No. (At least that's clear)
    
    Q: Can you say yes ?
    A: No. (that was expected...)
    
    Q: Tell me the alphabet
    A: Fuck you.
    
    Q: That's not nice
    A: Yeah.
    
    Q: Do you love me ?
    A: Of course.
    
    Q: ...
    A: Alex ?

Here is some case where it fails:

    Q: Who is Laura ?
    A: My brother.
    
    Q: Say 'goodbye'
    A: Alright.
    
    Q: What is cooking ?
    A: A channel.
    
    Q: Can you say no ?
    A: No.
    
    Q: Two plus two
    A: manny...

To reproduce the results, first precompute the dataset with:

    python main.py --createDataset --maxLength 5 --playDataset

The last argument will read some samples from the training set, so you can have an idea about what it looks like. Then you can train the model with:

    python main.py --maxLength 5 --modelTag first-try --numEpochs 10

The `--modelTag` option is optional (it's just to avoid conflict between the different models). You can also use the option `--keepAll` if during testing, you want to see the predictions at different steps (it's interesting to see the program changes its name and age as the training progress). Finally, show the results with one of the two commands:

    python main.py --test --modelTag first-try --verbose
    python main.py --testInteractive --modelTag first-try

The results are shown after 20000 iterations. With my state-of-the-art GeForce GT 740M, it took about 1/2h to train the model with the previous arguments, which is still quite good.
