# -*- coding: utf-8 -*-
# webpageclassifier.py

import math
import re
import numpy as np

from wpc_utils import *

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.estimator_checks import check_estimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

logging.basicConfig(level=logging.INFO)

"""Categorizes urls as blog|wiki|news|forum|classified|shopping|undecided.

THE BIG IDEA: It is inherently confusing to classify pages as clasifieds, blogs,
forums because of no single or clear definition. Even if there is a definition
the structure of the webpage can be anything and still comply with that definition.
The flow is very important for the categorization.

URL CHECK: The code checks the urls for WIKI, BLOGS, FORUMS and NEWS before anything
else. In case we have multiple clues in a single url such as www.**newsforum**.com,
it gives utmost precedence to the wiki. Then treats the others as equal and keeps
the result undecided hoping it will be decided by one of the successive processes.

WIKI: The easiest and most certain way of identifying a wiki is looking into its url.

BLOG: these mostly have a blog provider: And in most cases the name gets appended in the blog url itself.

FORUM: Although they come in different structure and flavors, one of the most
common and exact way of recognizing them is thru their:
    1. url: It may contain the word forum (not always true)
    2. html tags: the <table>, <tr>, <td> tags contains the "class" attribute that
       has some of the commonly repeting names like: views, posts, thread etc.
       The code not only looks for these exact words but also looks if these words
       are a part of the name of any class in these tags.

NEWS: Checking the <nav>, <header> and <footer> tags' data (attributes, text, sub tags
etc.) for common words we find in a news website like 'world', 'political', 'arts' etc
... 'news' as well and calculates the similary and uses it with a threshhold.

CLASSIFIED and SHOPPING: Here the code uses a two stage approch to first classify the
page into one of these using a list of words for each. The main difference assumed was
that a 'classified' page had many "touting" words, because it's people selling stuff,
whereas a 'shopping' page had different kinds of selling words (of course there is some
overlap, the the code takes care of that). Then it checks see if the predicted type is
independently relevent as a classified of shopping web page (using a threshhold).

The flow of how the sites are checked here is very important because of the heirarchy
on the internet (say a forum can be a shopping forum - the code will correctly classify
it as a forum)

The code uses some necessary conditions (if you may say) to find the accurate classification.
Checking the url, header and footer is also a very good	idea, but it may lead you astray
if used even before using the above mentioned accurate techniques. Especially the
words in the header and footer may lead you astray (say a footer may contain both 'blog'
and 'forum')

If indecisive this code will call the Hyperion Gray team categorizer
(That code is commented -- please also import their code first)

"""

LICENSE = """
Copyright [2015] [jpl]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

__author__ = ['Asitang Mishra jpl memex',
              'Charles Twardy sotera memex']

THRESH = 0.40
UNDEF = 'UNDEFINED'
ERROR = 'ERROR'
categories = 'blog, classified, forum, news, shopping, wiki'.split(', ')
categories.append(UNDEF)
gold_words = get_goldwords(categories, KEYWORD_DIR)


def ngrams(word, n):
    """Creates n-grams for a string, returning a list.
     :param word: str - A word or string to ngram.
     :param n: - n-gram length
     :returns: list of strings
    """
    ans = []
    word = word.split(' ')
    for i in range(len(word) - n + 1):
        ans.append(word[i:i + n])
    return ans


def url_has(url, wordlist):
    """True iff wordlist intersect url is not empty."""
    for word in wordlist:
        if word in url:
            return True
    return False


def cosine_sim(words, goldwords):
    """Finds the normalized cosine overlap between two texts given as lists.
    """
    # TODO: Speed up the loops? If profile suggests this is taking any time.
    wordfreq = collections.defaultdict(int)
    goldwordfreq = collections.defaultdict(int)
    commonwords = []
    cosinesum = 0
    sumgoldwords = 0
    sumwords = 0
    sqrt = math.sqrt

    for goldword in goldwords:
        goldwordfreq[goldword] += 1

    for word in words:
        wordfreq[word] += 1

    keys = wordfreq.keys()
    for word in goldwords:
        sumgoldwords += goldwordfreq[word] * goldwordfreq[word]
        if word in keys:
            commonwords.append(word)
            cosinesum += goldwordfreq[word] * wordfreq[word]

    for word in commonwords:
        sumwords += wordfreq[word] * wordfreq[word]

    logging.debug(commonwords)

    if sumgoldwords == 0 or sumwords == 0:
        return 0
    return cosinesum / (sqrt(sumwords) * sqrt(sumgoldwords))


def name_in_url(url):
    """Check for 'wiki', 'forum', 'news' or 'blog' in the url.
    'wiki' trumps; the rest must have no competitors to count.
    """
    count = 0
    if 'wiki' in url:
        return 'wiki'

    for word in ['forum', 'blog', 'news']:
        if word in url:
            url_type = word
            count += 1
    if count != 1:
        url_type = UNDEF
    return url_type


def forum_score(html, forum_classnames):
    """Return cosine similarity between the forum_classnames and
    the 'class' attribute of certain tags.
    """
    tags = ['tr', 'td', 'table', 'div', 'p', 'article']
    classlist = extract_all_classnames(tags, html)
    logging.debug(prettylist('forum classlist:', classlist))
    # Keep only matches, and only in canonical form. So 'forum' not 'forums'.
    # TODO: doesn't this artificially inflate cosine_sim? By dropping non-matches?
    classlist = [j for i in classlist for j in forum_classnames if j in i]
    logging.debug(prettylist('canonical form :', classlist))

    return cosine_sim(classlist, forum_classnames)


def news_score(html, news_list):
    """Check if a news website: check the nav, header and footer data
    (all content, class and tags within), use similarity
    """
    tags = ['nav', 'header', 'footer']
    contents = extract_all_fromtag(tags, html)
    contents = (re.sub('[^A-Za-z0-9]+', ' ', x.text).strip() for x in contents)
    contents = ' '.join(contents).split(' ')
    logging.debug(prettylist('news contents:', contents))
    return cosine_sim(contents, news_list)


def get_cosines(text, gold, vals={}):
    """Calculate all cosine similarity scores.
    :param text: - str, the HTML or text to score
    :param gold: - list, the list of gold words or key words
    :param vals: - dict of scores by name, including: forum, news, classified, shopping
    :returns: _vals_, overwriting those 4 fields, if supplied.
    
    """
    vals['forum'] = forum_score(text, gold['forum'])
    vals['news'] = news_score(text, gold['news'])
    text = re.sub(u'[^A-Za-z0-9]+', ' ', text)
    text_list = text.split(' ') + [' '.join(x) for x in ngrams(text, 2)]
    vals['classified'] = cosine_sim(text_list, gold['classified'])
    vals['shopping'] = cosine_sim(text_list, gold['shopping'])
    return vals


def make_jpl_clf(df,
                 categories: list,
                 goldwords: dict,
                 offline=False):
    """Create the JPL classifier with appropriate labels.
    :param df: 
    :param categories: 
    :param goldwords: 
    :param offline: 
    :return: the classifier, with cleaned labels in classifier.labels
    
    """
    logging.info("Creating JPL classifier")
    clf_pipe = Pipeline([#('stem', Lemmatizer()),
                         #('le', LabelEncoder()),
                         ('jpl', JPLPageClass(categories=categories,
                                              goldwords=goldwords,
                                              offline=offline))])
    pagetypes = list(df.pagetype)
    pagetypes = Lemmatizer(wnl=False).fit(pagetypes).transform(pagetypes)
    clf_pipe.fit(df.url, pagetypes)
    logging.info("Classifier 'training' completed.")
    return clf_pipe


class Lemmatizer(BaseEstimator, TransformerMixin):
    """Cleans up labels. Uses WordNet if avail, else simplistic strip final 's'.
    
    Based this on LabelTransf
    #TODO: Figure out how to get this into a classifier pipeline. 
    Right now it throws:
    `TypeError: fit_transform() takes 2 positional arguments but 3 were given`
    
    >>> labels = ['forum', 'forums', 'news', 'blog', 'blogs', 'news', 'error']
    >>> lem = Lemmatizer().fit(labels)
    >>> lem.classes_
    ['UNDEFINED', 'blog', 'error', 'forum', 'news']
    
    >>> lem.transform(labels)
    ['forum', 'forum', 'news', 'blog', 'blog', 'news', 'error']
        
    The wordnet lemmatizer fails here:
    >>> lem.transform(['wiki', 'wikis', 'shopping'])
    ['wiki', 'wikis', 'shopping']

    We want 'wikis' -> 'wiki'. So:
    >>> lem2 = Lemmatizer(wnl=False).fit(labels)
    >>> lem2.transform(['wiki', 'wikis', 'shopping'])
    ['wiki', 'wiki', 'shopping']
         
    """
    def __init__(self, goodlist=['news'], categories=[], wnl=True):
        """Create the lemmatizer.
        
        :param goodlist: list - words that bypass lemmatizer. In case WordNet unavail.
        :param categories: list - default category labels -- ensure they appear
        :param wnl: bool - Whether to use WordNet if available. 
        
        If WordNet is installed, uses a high-quality lemmatizer that probably doesn't
        need the `goodlist`.  Otherwise send `goodlist` to prevent the simple stripper 
        from turning "news" into "new", for example. 
        
        Note: `categories` will still be lemmatized unless in `goodlist`.   
        
        """
        if not wnl:
            self.wnl = False
        else:
            try:
                from nltk.stem import WordNetLemmatizer
                wnl = WordNetLemmatizer()
            except ImportError:
                wnl = False

        self.goodlist = frozenset(goodlist)
        self.categories = categories

    def fit(self, y):
        """'Fit' lemmatizer. Creates lemmatized classes_ list."""
        self.classes_ = sorted(set(self._transform(y + self.categories)))
        return self

    @classmethod
    def _stem(self, word):
        ans = word.strip()
        if ans[-1] != 's' or ans[-2] == 's':
            return ans
        return ans[:-1]

    def _transform(self, y):
        """Lemmatize the labels."""
        goodlist = self.goodlist
        stem = self._stem
        if self.wnl:
            stem = self.wnl.lemmatize

        ans = list(y)
        for i, word in enumerate(y):
            if word in goodlist:
                continue
            ans[i] = stem(word)
        return ans

    def transform(self, y):
        """Lemmatize the labels and check if new labels have appeared."""
        labels = self._transform(y)
        classes = np.unique(labels)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            logging.warning("Added new labels from y: %s" % str(diff))
            self.classes_ = np.append(self.classes_, classes)

        return labels


class JPLPageClass(BaseEstimator):
    """Classify pagetype based on the URL and HTML. Tries fastest first.
    
    Feed it URLs. If it can decide on those, it does, else it uses 
    `get_html()` to fetch HTML and try other methods. 
    
    Categorizes urls as blog | wiki | news | forum | classified | shopping | _UNDEF_.
    Returns best guess and a dictionary of scores, which may be empty.
    
    """

    def __init__(self,
                 goldwords: dict,
                 offline=False,
                 thresh=0.40,
                 categories: list=[UNDEF, ERROR]):
        """Set up the JPL Page Classifier.
        
        :param goldwords: dict {label -> "golden words" related to that category
        :param offline: bool - True if HTML can be found in standard file loc'n
        :param thresh: float - If no category > thresh, return UNDEF.
        :param categories: these are the labels we expect we might see

        
        Note: stored categories will be _singular_ --> no trailing 's'
                
        """
        self.goldwords = goldwords
        self.offline = offline
        self.thresh = thresh
        self.categories = categories
        self.bleached = []
        self.errors = []
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """Not really fitting, but...
        
        :param X: list[str] - a list of URLs
        :param y: list[str] - a list of page types or classes
        
        """

        self.classes_= np.unique(list(y) + self.categories)
        self.labels = y
        logging.info('\tclasses_: %s' % self.classes_)
        #X, y = check_X_y(X, y)
        return self

    def _score_url(self,
                   url: 'The URL ',
                   bleach: bool):
        """Score the URL using JPL cascade. Only load HTML if URL inconclusive.

        :param url: The url to score 
        :param bleach: bool - Try cleaner url if original fails.
        :return: score vector (numpy array)
        """
        # TODO: Move definitions back to predict_proba, and pass in.
        # As-is, it has to do "self." lookups for each URL.
        logging.info('URL: %s' % url[7:MAX_URL_LEN])
        𝜃 = self.thresh
        scores = np.ones(len(self.classes_)) * .1
        idx = dict([(key,i) for i, key in enumerate(self.classes_)])
        def tally(key, val):
            scores[idx[key]] = val

        # 1. Check for blog goldwords in URL
        if url_has(url, self.goldwords['blog']):
            tally('blog', .9)
        else:
            # 2. Check for category name in URL
            name_type = name_in_url(url)
            if name_type != UNDEF:
                tally(name_type, .9)
        if max(scores) > 𝜃:
            return scores / sum(scores)

        # TODO: URL ngrams

        # 3. Look at the HTML.
        html = get_html(url, offline=self.offline)
        if html.startswith(HTTP_ERROR):
            if bleach and bleach_url(url) != url:
                logging.info("Bleaching URL...")
                self.bleached.append(url)
                return self._score_url(bleach_url(url), bleach=False) # Avoid inf loop!
            else:
                logging.warning('%s: %s ' % (HTTP_ERROR, clean_url(url)))
                self.errors.append(url)
                tally(ERROR, .9)
        else:
            vals = get_cosines(html, self.goldwords)
            for key, val in vals.items():
                tally(key, val)
        if max(scores) > 𝜃:
            return scores / sum(scores)

        # Fallback
        tally(UNDEF, 1 - max(scores))
        return scores / sum(scores)

    def predict(self, X):
        """Return the most likely class, for each x in X."""
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]

    def predict_proba(self, X):
        """For each x in X, provide vector of probs for each class.
        
        Assume X is a list of URLs, and that `get_html(url)` will
        retrieve HTML as required.
        
        """
        #check_is_fitted(self, ['X_', 'y_'])
        #X = check_array(X)
        surl = self._score_url
        return [surl(url, bleach=True) for url in X]


#check_estimator(JPLPageClass)
if __name__ == "__main__":
    import pandas as pd

    URL_FILE = '../thh-classifiers/dirbot/full_urls.json'
    # URL_FILE = '50urls.csv'
    SCORE_FILE = 'scores.csv'
    ERR_FILE = 'url_errs.json'
    OFFLINE = True
    MAX_N = 500
    MAX_URL_LEN = 70

    df = pd.read_json(URL_FILE)
    if OFFLINE:
        logging.info("Running in OFFLINE mode.")
    logging.info("\n%s\nwebpageclassifier\n%s" % ('-'*18,'-'*18))

    df = df.sample(n=MAX_N, random_state=42)  # Subset for testing
    clf = make_jpl_clf(df,
                       categories=[UNDEF, ERROR],
                       goldwords=gold_words,
                       offline=True)
    #probs = clf.predict_proba(df)
    predicted = clf.predict(df.url)
    labels = clf.named_steps['jpl'].labels

    # Show each result
    for url, cat in zip(df.url, predicted):
        print('%25r => %s' % (url[7:30], cat))

    # Remove rows with errors
    labels, predicted = zip(*[row for row in zip(labels, predicted) if row[1] != ERROR])
    labels, predicted = np.array(labels), np.array(predicted)
    classes_ = list(clf.classes_)
    classes_.remove(ERROR)

    # Metrics
    print(metrics.classification_report(labels, predicted))
    print("Confusion Matrix:")
    for row in zip(classes_, metrics.confusion_matrix(labels, predicted)):
        print('%20s: %s' % row)
    print("\n   µ Info: %4.2f" % metrics.adjusted_mutual_info_score(labels, predicted))
    # Homebrew reporting
    model = clf.steps[-1][1]
    print('  Total #: %4d' % len(df.url))
    print('  #Errors: %4d \t(%4d Bleached)' % (len(model.errors), len(model.bleached)))
    print('#Predictd: %4d' % len(predicted))
    print(' Accuracy: %4.2f' % np.mean(predicted == labels))
    print("\nErrors:")
    for row in model.errors:
        print('\t', row)
    #print(metrics.auc(labels, predicted))
    #print("ROC Curve:")
    #print(metrics.roc_curve(labels, predicted))

    #df, df_err, report = score_df(df, answers, probs)
    #df.to_csv(SCORE_FILE)  # , float_format='5.3f')
    #df.to_json(ERR_FILE)
    #print("\nURLs with Errors\n---------------\n", df_err)
    #print("Errors also saved to", ERR_FILE)
    #print(report)
