#!/usr/bin/env python3
"""
tweet classification for B551 homework 2

Team members:
Hai Hu, Yuhan Zeng


First detect encoding of tweets:

```
chardet3 tweets.train.txt
```

turns out to be: ISO-8859-2

"""

import glob, math, sys
from collections import Counter
from stopwords import stopwords

# mamually define unwanted words
unwanted = {
    '\n', '-', "i'm", '&amp;', '__', '/', '___', '____',
    'job', 'jobs', 'hiring', 'hiring\n', 'b/w',
    '*',
}

stopwords.update(unwanted)

def main():
    if len(sys.argv) != 4:
        print('\nusage: ./geolocate.py training-file testing-file output-file\n')
        exit()

    # ./geolocate.py tweets.train.clean.txt tweets.test1.clean.txt output.txt

    fn_train = sys.argv[1]
    fn_test = sys.argv[2]
    fn_out = sys.argv[3]

    # top_n=20000, thresh=5 acc: 0.57
    model = Model(fn_train, top_n=10000, thresh=5)
    # model = Model(fn_train, fn_test, fn_out, top_n=0, thresh=5, alpha=.5)
    model.predictFile(fn_test, fn_out)

    print()
    top_n = 5
    for city in sorted(model.vocab.keys()):
        print('top 5 features for', city, end=' : ')
        print(reversed([x[0] for x in \
               sorted(model.vocab_p[city].items(),
                      key = lambda x:x[1])[-top_n:]]))

class Model():
    """ 
    a naive bayes model

    for a test tweet T: w1 w2

    P(T=Chicago | w1, w2)
    = P(w1, w2 | Chicago) * P(Chicago) / P(w1, w2)
    = argmax P(w1, w2 | Chicago) * P(Chicago)              [drop denominator]
    = argmax P(w1|Chicago) * P(w2|Chicago) * P(Chicago)    [independence assumption]

    Thus, we need to estimate:

    P(w_i|Chicago) for all words w_i, and P(Chicago)

    """
    def __init__(self, fn_train, top_n=0, thresh=5, alpha=None):
        self.thresh = thresh       # threshold to ignore feats
        self.cities = set()        # 12 in train
        self.vocab = {}            # {city1: {word1:10, word2:20} ...}
        self.V = {}                # all vocab of train {word:count}
        self.vocab_p = {}          # probs of vocab
        self.probCity = {}         # probs for each city
        self.top_n = top_n         # top_n = 5, 10 ...
        self.top_n_words = set()   # top n features for each lang in one set
        self.sums = {}             # { city1: |V_city1| ... }
        self.alpha = alpha         # alpha in add-1 smoothing
        self.countVectorize(fn_train)
        self.getProbs()

    def cleanTweet(self, tweet):
        """
        clean tweet, which is a string
        return a list of features
        """
        tweet = tweet.lower().replace('#', '').replace('@', '').replace('\n', '')
        words = tweet.split(' ')
        newtweet = []

        # word unigram
        newtweet += [
            word \
            for word in words \
            if word not in stopwords
            ]

        # word bigram
        # newtweet += [
        #     '_'.join(words[i:i+2])
        #     # word \
        #     for i in range(len(words)-1) \
        #     # if word not in stopwords
        #     ]

        # character n-gram
        n = 4
        for m in range(3, n+1):
            newtweet += [ tweet.lower()[i:i+m] \
                          for i in range(0, len(tweet)-m-1) \
                          if tweet.lower()[i:i+m] not in stopwords]

        return newtweet

    def getTopN(self):
        """
        get top n features for each city
        """
        for city in sorted(self.vocab.keys()):
            self.top_n_words.update([x[0] for x in \
                   sorted(self.vocab[city].items(), key=lambda x: x[1])[-self.top_n:]])

    def countVectorize(self, fn_train):
        """ get table of word frequencies, like countVectorize in scikit learn
                 w1   w2   w3
        city1    10   20   5
        city2  ...
        """
        # fn_train = 'tweets.train.txt'
        cities = {}  # counts: cities[city1] = 10; cities[city2] = 20
        with open(fn_train) as f:
            for line in f:
                line_lst = line.split(' ')
                city = line_lst[0]
                if ',_' in city:  # handle multi-line tweets
                    self.cities.add(city)
                    if city not in self.vocab:
                        self.vocab[city] = {}  # add city
                        cities[city] = 1
                    else:
                        cities[city] += 1
                else:
                    print('something wrong reading train')
                    exit()

                # get feats
                tweet = ' '.join(line_lst[1:])
                tweet = self.cleanTweet(tweet)

                # add to vocab
                for word in tweet:
                    if word == '': continue
                    self.vocab[city][word] = self.vocab[city].get(word, 0) + 1

        # get top_n features for each city
        if self.top_n:
            self.getTopN()

        # vocabulary V
        V = set( [word for city in self.vocab.keys() \
                            for word in self.vocab[city].keys()] )
        self.V = { word: sum( self.vocab[city].get(word,0) \
                              for city in self.vocab.keys() ) \
                              for word in V}
        print('\nlen V before:', len(self.V))

        # filter according to threshold
        self.V = { word: self.V[word] for word in self.V.keys() \
                   if self.V[word] > self.thresh}
        print('len V after thresh:', len(self.V))

        # filter according to whitelist
        if self.top_n_words:
            self.V = { word: self.V[word] for word in self.V.keys() \
                       if word in self.top_n_words}
        print('len V after top_n:', len(self.V))

        # probs of cities
        self.probCity = { city: cities[city] / sum(cities.values()) for city in cities }
        try: assert sum(self.probCity.values()) == 1
        except AssertionError: print(sum(self.probCity.values()))

        # sanity check
        # print(self.cities)
        # print([len(x) for x in self.vocab.values()])  # |V| for each city
        # print(sum( [len(x) for x in self.vocab.values()] ))  # absolute sum
        # assert self.vocab['Manhattan,_NY']['Tra'] == 4
        # for k, v in reversed(sorted(self.vocab['Chicago,_IL'].items(), key = lambda x: x[1])):
        #     print(k, v)

    def getProbs(self, smoothing=None):
        """
        convert counts to probs
        smoothing = NoSmoothing, addOne
        """
        for city in self.vocab.keys():
            self.sums[city] = sum([self.vocab[city][word] \
                              for word in self.vocab[city].keys() \
                              if word in self.V ])
        # print(sorted(sums.items(), key = lambda x:x[0]))
        for city in self.vocab.keys():
            # add-1 smoothing TODO
            if self.alpha:
                self.vocab_p[city] = { word: (self.vocab[city][word]+self.alpha) / \
                                             (self.sums[city]+self.alpha*len(self.V)) \
                                       for word in self.vocab[city].keys() \
                                       if word in self.V}
            else:
                # vanilla NB
                self.vocab_p[city] = { word: self.vocab[city][word]/self.sums[city] \
                                       for word in self.vocab[city].keys() \
                                       if word in self.V}

    def predict(self, tweet):
        """
        give the city w/ highest probs
        tweet: a string "this is my tweet"

        - add log(prob) for each word, instead of multiplying
        """
        tweet = self.cleanTweet(tweet)
        logprobs = {}   # logprobs for each city

        # TODO workaround for smoothing
        # min prob for each city
        mins = { city:min(self.vocab_p[city].values()) for city in self.vocab_p.keys() }

        for city in self.vocab_p.keys():
            ### key step ###
            # term1        + term2
            # log(P(city)) + sum(P(word|city))
            term1 = math.log2(self.probCity[city])

            # only when the word is known in vocab[city]
            term2 = 0
            for word in tweet:
                if word in self.vocab_p[city].keys():
                    term2 += math.log2(self.vocab_p[city][word])
                else:  # out of vocab
                    # add-1 smoothing TODO
                    if self.alpha:
                        term2 += math.log2(self.alpha / self.sums[city] + len(self.V))
                    else:
                        # vanilla NV: now add p of most infrquent word
                        term2 += math.log2(mins[city])

            logprobs[city] = term1 + term2

        for city, logprob in sorted(logprobs.items(), key = lambda x: x[1]):
            # print(city, logprob)
            ans = city
            pass
        return ans  # San_Francisco,_CA

    def predictFile(self, fn_test, fn_out):
        """
        predict test file
        """
        test_strs = []
        gold = []
        pred = []  # prediction
        with open(fn_test) as f:
            for line in f:
                line_lst = line.split(' ')
                city_tmp = line_lst[0]
                if city_tmp in self.vocab.keys():  # a city
                    test_strs.append(' '.join(line_lst[1:]))
                    gold.append(city_tmp)
                else:
                    print('something wrong reading test file')
                    exit()

        # print(''.join(test_strs))

        for tweet in test_strs:
            ans = self.predict(tweet)
            pred.append(ans)

        # accuracy
        assert len(gold) == len(pred) == len(test_strs)
        acc = sum( [1 for idx in range(len(gold)) if gold[idx] == pred[idx]] ) / \
            len(gold)
        print('*** accuracy:', acc)

        # majority baseline
        # print(Counter(gold))
        print( [ (  'majority baseline:', x[0], x[1],
                    x[1] / sum(Counter(gold).values()) ) \
                 for x in Counter(gold).items() \
                 if x[1] == max( Counter(gold).values() )
                 ])

        # write output
        with open(fn_out, 'w') as f:
            for i in range(len(gold)):
                f.write("{} {} {}".format(pred[i], gold[i], test_strs[i]))

if __name__ == '__main__':
    main()


