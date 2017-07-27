import json
import os
import re
import itertools

from TwitterAPI import TwitterAPI

with open("chatbot/credentials.json") as f:
    credentials = json.load(f)

api = TwitterAPI(**credentials)


def get_tweets(screen_name, max_tweets=None):
    show = api.request("users/show", {"screen_name": screen_name}).json()
    max_tweets = max_tweets or show.get("statuses_count")
    max_tweets = min(max_tweets, 3200)
    print("Gathering {} tweets. Through API, 3200 is max possible".format(max_tweets))
    user_tweets = []
    query_params = {"screen_name": screen_name, "max_id": None, "count": 200}
    last_seen = True
    print("Gathering tweets for", screen_name)
    while True:
        try:
            r = api.request("statuses/user_timeline", query_params)
            timeline_tweets = r.json()
            if timeline_tweets[-1]['id'] == last_seen:
                break
            last_seen = timeline_tweets[-1]['id']
            user_tweets.extend(timeline_tweets)
            query_params['max_id'] = timeline_tweets[-1]['id']
            print("latest ID", query_params['max_id'],
                  "number of new tweets", len(timeline_tweets))
        except Exception as e:
            print("ERROR, check twitter handle", e)
        if len(user_tweets) >= max_tweets:
            break
    seen = set()
    tweets = []
    for x in user_tweets:
        if x['id'] not in seen:
            tweets.append(x)
            seen.add(x['id'])
    return tweets


def find_questions_for_tweets(tweets):
    origins = {tweet['in_reply_to_status_id']: tweet
               for tweet in tweets if tweet.get('in_reply_to_status_id')}
    origin_gen = (x for x in origins)
    questions = []
    answers = []
    print("Getting original tweets to which <user> replied")
    while True:
        orig = list(itertools.islice(origin_gen, 100))
        if not orig:
            break
        id_query = ",".join([str(x) for x in orig])
        orig_tweets = api.request("statuses/lookup", {"id": id_query}).json()
        for ot in orig_tweets:
            if ot['id'] in origins:
                questions.append(ot['text'])
                answers.append(origins[ot['id']]['text'])
        print("collected question/answer pairs", len(questions), len(answers))
    return questions, answers


def normalize_tweet(x):
    x = " ".join(x.split())
    x = x.lower()
    x = re.sub("http[^ ]+", "LINK", x)
    x = re.sub("#[^ ]+", "TAG", x)
    x = re.sub("(@[^ ]+ )*@[^ ]+", "MENTION", x)
    for punc in [".", ",", "?", "!"]:
        x = re.sub("[{}]+".format(punc), " " + punc, x)
    x = x.replace("n't", " not")
    x = " ".join(x.split())
    x = x.lstrip("MENTION ")
    return x.strip()


def get_tweet_qa(twitter_username, max_tweets=None, normalize_tweets=True):
    tweets = get_tweets(twitter_username, max_tweets)
    questions, answers = find_questions_for_tweets(tweets)
    if normalize_tweets:
        questions = [normalize_tweet(x) for x in questions]
        answers = [normalize_tweet(x) for x in answers]
    return questions, answers


def get_rate_limits():
    rates = api.request("application/rate_limit_status").json()
    timeline = rates['resources']['statuses']['/statuses/user_timeline']
    lookup = rates['resources']['users']['/users/lookup']
    print("lookup", lookup)
    print("timeline", timeline)
    return timeline['remaining'] != 0 and lookup['remaining'] != 0


def store_question_answers(username, max_number=None):
    questions, answers = get_tweet_qa(username, max_number)
    d = "data/tweets/" if os.path.isdir("data/tweets") else "../data/tweets/"
    d += "{}-{}.txt"
    with open(d.format(username, "questions"), "w") as f:
        f.write("\n".join(questions))
        print("Saved", d.format(username, "questions"))
    with open(d.format(username, "answers"), "w") as f:
        f.write("\n".join(answers))
        print("Saved", d.format(username, "answers"))
    return questions, answers