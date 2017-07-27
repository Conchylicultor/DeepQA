import os
import pickle

from chatbot.corpus.twitter.deep_twitter_qa import store_question_answers


def genset(questions, answers, maxLength):
    words = set()
    questions = [x.split()[-maxLength:] for x in questions]
    answers = [x.split()[:maxLength] for x in answers]
    for x in questions:
        words.update(x)
    for x in answers:
        words.update(x)
    words = sorted(words)
    word2id = {x: num + 4 for num, x in enumerate(words)}
    id2word = {num + 4: x for num, x in enumerate(words)}
    specials = ['<pad>', '<go>', '<eos>', '<unknown>']
    id2word.update({num: x for num, x in enumerate(specials)})
    word2id.update({x: num for num, x in enumerate(specials)})
    trainingSamples = []
    for q, a in zip(questions, answers):
        trainingSamples.append([[word2id[x] for x in q],
                                [word2id[x] for x in a]])
    return {"trainingSamples": trainingSamples,
            "id2word": id2word,
            "word2id": word2id}


def load_qa(username, max_tweets, overwrite):
    d = "data/tweets/" if os.path.isdir("data/tweets") else "../data/tweets/"
    d += "{}-{}.txt"
    if overwrite:
        questions, answers = store_question_answers(username, max_tweets)
    else:
        try:
            with open(d.format(username, "questions")) as f:
                questions = [x for x in f.read().split("\n")]
                print("Loaded", d.format(username, "questions"))
            with open(d.format(username, "answers")) as f:
                answers = [x for x in f.read().split("\n")]
                print("Loaded", d.format(username, "answers"))
        except FileNotFoundError:
            print("QA files not found, downloading.")
            questions, answers = store_question_answers(username, max_tweets)
    return questions, answers


def TwitterData(username, maxLength=10, max_tweets=3200, filterVocab=1, vocabSize=40000, overwrite=False):
    sample_path = "data/samples/twitter/"
    path = os.path.join(sample_path + '{}-dataset-length{}-filter{}-vocabSize{}.pkl'.format(username, maxLength, filterVocab, vocabSize))
    if not overwrite and os.path.isfile(path):
        print("Using existing pickle", path)
    else:
        questions, answers = load_qa(username, max_tweets, overwrite)
        data = genset(questions, answers, maxLength)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        msg = "Saved model. {} unique words found. {} QA pairs created."
        print(msg.format(len(data['id2word']), len(questions)))