# Deep Q&A

## Presentation

This work tries to reproduce the results of [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It uses a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

The loading corpus part of the program is inspired by the Torch [neuralconvo](https://github.com/macournoyer/neuralconvo) from [macournoyer](https://github.com/macournoyer).

For now, it uses the [Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) corpus, but one of the long-term goal is to test it on bigger corpus.

## Installation

The program requires the following dependencies (easy to install using pip):
 * python 3
 * tensorflow (tested with v0.9.0)
 * numpy
 * CUDA (for using gpu, see TensorFlow [installation page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) for more details)
 * nltk (natural language toolkit for tokenized the sentences)
 * tqdm (for the nice progression bars)

The Cornell dataset is already included.

The web interface requires some additional packages:
 * django (tested with 1.10)
 * channels
 * Redis (see [here](http://redis.io/topics/quickstart))
 * asgi_redis

## Running

### Chatbot

To train the model, simply run `main.py`. Once trained, you can test the results with `main.py --test` (results generated in 'save/model/samples_predictions.txt') or `main.py --test interactive` (more fun).

Here are some flags which could be useful. For more help and options, use `python main.py -h`:
 * `--modelTag <name>`: allow to give a name to the current model to differentiate between them when testing/training.
 * `--keelAll`: use this flag when training if when testing, you want to see the predictions at different steps (it can be interesting to see the program changes its name and age as the training progress). Warning: It can quickly take a lot of storage space if you don't increase the `--saveEvey` option.
 * `--verbose`: when testing, will print the sentences as they are computed.
 * `--playDataset`: show some dialogue samples from the dataset (can be use conjointly with `--createDataset` if this is the only action you want to perform).

To visualize the computational graph and the cost with [TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html), just run `tensorboard --logdir save/`.

By default, the network architecture is a standard encoder/decoder with two LSTM layers (hidden size of 256) and an embedding size for the vocabulary of 32. The network is trained using ADAM. The maximum sentence length is set to 10 words, but can be increased.

### Web interface

Once trained, it's possible to chat with it using a more user friendly interface. The server will look at the model present on `save/model-server/model.ckpt`. The first time you want to use it, you'll need to configure it with:

```bash
export CHATBOT_SECRET_KEY "my-secret-key"
cd chatbot_website/
python manage.py makemigrations
python manage.py migrate
```

Then, to launch the server locally, use the following commands:

```bash
cd chatbot_website/
redis-server &
python manage.py runserver
```

After launch, the interface should be available on [http://localhost:8000/](http://localhost:8000/).

[![Chatbot web interface](http://e-pot.xyz/cv/thumb_chatbot.png)](http://e-pot.xyz/cv/chatbot.png)

## Results

Surprisingly, it's possible to get some results after only 1 or 2 hours of training (on a GeForce GT 740M), by drastically reducing the input sequence to 5 words and output to 3 (plus the `go` and `eos` tokens) and by using a small embedding size (something like 25). Since then I modified the code and now the output length has to match the input one but you can still reproduce the original results, using [this version](https://github.com/Conchylicultor/DeepQA/tree/92863e2929580818f866f16969909fe2093d41d1).
Of course, the network won't be really chatty:

    Q: Hi
    A: Hi.
    
    Q: What is your name ?
    A: Laura.
    
    Q: What does that mean ?
    A: I dunno.
    
    Q: How old are you ?
    A: thirty-five.
    
    Q: Will Google hire me ?
    A: No.
    
    Q: Tell me the alphabet
    A: Fuck you.
    
    Q: That's not nice
    A: Yeah.

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
    A: Manny...

With longer sentences, the network is much slower to train. After 250000 iterations with maxLength=10, the network still mainly gives evasive answers (by answering most of the open ended questions by 'I don't know') but it seems to have integrated some basic semantic:

    Q: When are the fireworks ?
    A: Tomorrow.
    
    Q: What color ?
    A: Gray.
    
    Q: Which apple ?
    A: That's the first one.
    
    Q: What time is it ?
    A: Eleven o'clock.
    
    Q: Who is Skywalker ?
    A: She's a freelance bio-exorcist.
    
    Q: Who is John ?
    A: The old man.
    
    Q: Who is Jenny ?
    A: The man who killed herself.

    Q: When were you born ?
    A: Soon.

As shown on the last questions, the network still does some confusion between the genders and the present/future. I also tried some deeper philosophical questions with more or less success.

    Q: What is the purpose of being intelligent ?
    A: The man are obsolete.
    
    Q: What is immoral ?
    A: The truth.
    
    Q: What is morality ?
    A: The worst.
    
    Q: What is the purpose of existence ?
    A: It 's private.
    
    Q: What happens if machines can think ?
    A: I think I'll just lie with you.

The model I trained is quite limited by the embedding size I put and by its size, and by the training corpus size. Its 'thought vector' is also probably too small to answer the kind of following:

    Q: My name is David. What is my name ?
    A: Debbie.
    
    Q: My name is John. What is my name ?
    A: Carl Smith.
    
    Q: How many legs does a cat have ?
    A: Four.
    
    Q: How many legs does a spider have ?
    A: Four.
    
    Q: Do you prefer football or soccer ?
    A: Whiskey.

It also seems to overfit as sometimes it will just pop out sentences from its training set unrelated to the question. Adding some dropout should reduce the issue.

    Q: I play tennis. What do I play ?
    A: I'm going to get laid.
    
    Q: What color is the sky ?
    A: One of the deputies is everywhere.

You can find a pre-trained model [here](https://drive.google.com/file/d/0Bw-phsNSkq23amlSZXVqcm5oVFU/view?usp=sharing). Don't expect spectacular results though. If you have a high-end GPU, you could try to increase the network parameters and train a better model.
