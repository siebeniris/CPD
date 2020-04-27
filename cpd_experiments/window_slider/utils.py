import string
import time

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")


def lemmatize_sent(sent):
    relate = []
    doc = nlp(sent)

    # get only lemmas and non entities.
    for token in doc:
        if token.text not in STOP_WORDS:
            if token.pos_ in ['PROPN', 'NOUN', 'VERB', 'ADJ'] and not token.is_punct:
                t = token.lemma_.translate(str.maketrans('', '', string.punctuation)).strip()
                if len(t) > 1:
                    relate.append(t)
    return relate