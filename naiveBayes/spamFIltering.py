from cmath import exp, log
import re
import string
import pandas as pd
import csv
import itertools
from sklearn.model_selection import train_test_split

def tokenize(text: str) -> set[str]:
    words: list[str] = []
    for word in re.findall(r'[A-Za-z0-9\']+', text):
        words.append(word.lower())
    return set(words)

df = pd.read_csv(r"C:\Users\sikor\naiveBayes\archive\spamDataset.csv")

X = list(list(df["Message"]))

Y = list(df["Category"])

flat_map = lambda f, xs: sum(map(f, xs), [])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

class Message:
    def __init__(self, message, isSpam):
            self.message = message
            self.isSpam= isSpam

class NaiveBayes:
    def __init__(self, k=1) -> None:
        self._k: int = k
        self._num_spam_messages: int = 0
        self._num_ham_messages: int = 0
        self._num_word_in_spam: dict[int] = dict()
        self._num_word_in_ham: dict[int] = dict()
        self._spam_words: set[str] = set()
        self._ham_words: set[str] = set()
        self._words: set[str] = set()

    def train(self, messages: list[Message]) -> None:
        msg: Message
        token: str
        for msg in messages:
            tokens: set[str] = tokenize(msg.message)
            self._words.update(tokens)
            if msg.isSpam:
                self._num_spam_messages += 1
                self._spam_words.update(tokens)
                for token in tokens:
                    if (token not in self._num_word_in_spam):
                        self._num_word_in_spam[token] = 0
                    self._num_word_in_spam[token] += 1
            else:
                self._num_ham_messages += 1
                self._ham_words.update(tokens)
                for token in tokens:
                    if (token not in self._num_word_in_ham):
                        self._num_word_in_ham[token] = 0
                    self._num_word_in_ham[token] += 1

    #P(w|Spam)    
    def _p_word_spam(self, word: str) -> float:
        numWordInSpam = self._num_word_in_spam[word] if word in self._num_word_in_spam else 1 # number of this word repeats in spam
        return (self._k + numWordInSpam) / ((2 * self._k) + self._num_spam_messages) # word repeats in spam / spam messages count

    def _p_word_ham(self, word: str) -> float:
        numWordInHam = self._num_word_in_ham[word] if word in self._num_word_in_ham else 1
        return (self._k + numWordInHam) / ((2 * self._k) + self._num_ham_messages)

    # P(class|Message) = P(class) * Product(P(word | Spam))
    def isSpamMessage(self, text: str) -> bool:
        text_words: set[str] = tokenize(text)
        log_p_spam: float = 0.0
        log_p_ham: float = 0.0

        for word in self._words:
            p_spam: float = self._p_word_spam(word)
            p_ham: float = self._p_word_ham(word)
            if word in text_words:
                log_p_spam += log(p_spam) # to avoid arithmetic underflow
                log_p_ham += log(p_ham) # log(ab) = log a + log b
            else:
                log_p_spam += log(1 - p_spam)
                log_p_ham += log(1 - p_ham)

        p_if_spam: float = exp(log_p_spam) # 'unwrap' log
        p_if_ham: float = exp(log_p_ham)
        # return p_if_spam / (p_if_spam + p_if_ham) # normalize to 1

        return p_if_spam.real >= p_if_ham.real

messagesTrain = [Message(xy[0], xy[1] == 'spam') for xy in itertools.zip_longest(x_train, y_train)]

bayesClassifier = NaiveBayes()
bayesClassifier.train(messagesTrain)

classMatching = 0
for xy in itertools.zip_longest(x_test, y_test):
    isSpamPrediction = bayesClassifier.isSpamMessage(xy[0])
    print("Is Spam predict: " + str(bayesClassifier.isSpamMessage(xy[0])) + ", and actually is: " + str(xy[1]))
    if isSpamPrediction & (str(xy[1]) == "spam"):
        classMatching+=1
    elif isSpamPrediction == False & (str(xy[1]) == "ham"):
        classMatching+=1

print ("Accuracy: " + str(classMatching) + " / " + len(y_test))

input()