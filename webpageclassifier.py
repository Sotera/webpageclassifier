# -*- coding: utf-8 -*-
# webpageclassifier.py

import math
import re
import numpy as np
import arrow

from wpc_utils import *

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib.parallel import cpu_count, Parallel, delayed
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
from pandas import DataFrame, Series

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

ERROR: Uses goldenwords from a labeled set: words that occurred on error pages but not
 others.  Think '404' and less diagnostic sets. Tried last on the assumption an error
 page won't resemble other things. 

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
UNDEF = 'UNCERTAIN'
ERROR = 'error'
OFFLINE = True
MAX_URL_LEN = 70  # For pretty-printing

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


def get_contents(tags, html, label):
    """Extract _tags_ from _html_, normalize, and return as string of words"""
    contents = extract_all_fromtag(tags, html)
    contents = (re.sub('[^A-Za-z0-9]+', ' ', x.text).strip() for x in contents)
    logging.debug(prettylist('%s contents:' % label, contents))
    return ' '.join(contents).split(' ')

def forum_score(html, forum_classnames):
    """Return cosine similarity between the forum_classnames and
    the 'class' attribute of certain tags.
    """
    tags = ['tr', 'td', 'table', 'div', 'p', 'article']
    contents = get_contents(tags, html, 'forum')
    # Keep only matches, and only in canonical form. So 'forum' not 'forums'.
    # TODO: doesn't this artificially inflate cosine_sim? By dropping non-matches?
    contents = [j for i in contents for j in forum_classnames if j in i]
    logging.debug(prettylist('canonical form :', contents))
    return cosine_sim(contents, forum_classnames)


def news_score(html, news_list):
    """Check if a news website: check the nav, header and footer data
    (all content, class and tags within), use similarity
    """
    tags = ['nav', 'header', 'footer']
    contents = get_contents(tags, html, 'news')
    return cosine_sim(contents, news_list)


def error_score(html, error_list):
    """Check text against ERROR goldenwords; use similarity
    """
    tags = ['tr', 'td', 'table', 'div', 'p', 'article', 'body']
    contents = get_contents(tags, html, ERROR)
    return cosine_sim(contents, error_list)

def get_cosines(text, gold, vals={}):
    """Calculate all cosine similarity scores.
    :param text: - str, the HTML or text to score
    :param gold: - dict, the dict of gold word lists, keyed by category
    :param vals: - dict of scores by name, including: forum, news, classified, shopping
    :returns: _vals_, overwriting those 4 fields, if supplied.
    
    """
    vals['forum'] = forum_score(text, gold['forum'])
    vals['news'] = news_score(text, gold['news'])
    text = re.sub(u'[^A-Za-z0-9]+', ' ', text)

    text_list = text.split(' ')
    vals[ERROR] = cosine_sim(text_list, gold[ERROR])
    vals['shopping'] = cosine_sim(text_list, gold['shopping'])

    text_list.extend([' '.join(x) for x in ngrams(text, 2)])
    vals['classified'] = cosine_sim(text_list, gold['classified'])
    return vals


def make_jpl_clf(X, y,
                 goldwords: dict=None,
                 offline=True):
    """Wrapper to create the JPL classifier.
     
     Note that "fitting" this classifier is just checking for extraneous labels (y).
     Previously I had tried to put it in a pipeline with a Lemmatizer to clean
     up the labels. But that doesn't work because pipeline.transform() affects X, 
     not y.  So it's up to the caller to notice the mismatch and fix. 
     
     However, despite having only a single step, I'll leave the Pipeline in place,
     in case we later find it useful to preprocess the URLs.
    
    :param X: Input data, here **urls**.
    :param y: Labels -- only used to check for unexpected values.
    :param goldwords: dict of {category: wordlist} of gold words for each category.
    :param offline: set True to use cached pages in KEYWORD_DIR
    :param xcol: name of column containing URLs
    :param ycol: name of column containing labels
    :return: the classifier
    
    """
    logging.info("Creating JPL classifier")
    clf_pipe = Pipeline([('jpl', JPLPageClass(goldwords=goldwords,
                                              offline=offline))])
    clf_pipe.fit(X, y)
    logging.info("Classifier 'training' completed.")
    return clf_pipe


class Lemmatizer(BaseEstimator, TransformerMixin):
    """Cleans up labels. Uses WordNet if avail, else simplistic strip final 's'.
    
    Based this on LabelTransform: it CANNOT be put into a classifier pipeline,
    because it transforms y, not X.
    
    >>> labels = ['forum', 'forums', 'news', 'blog', 'blogs', 'news', 'error']
    >>> lem = Lemmatizer().fit(labels)
    >>> lem.classes_
    ['blog', 'error', 'forum', 'news']
    
    >>> lem.transform(labels)
    ['forum', 'forum', 'news', 'blog', 'blog', 'news', 'error']
        
    The wordnet lemmatizer fails here:
    >>> lem.transform(['wiki', 'wikis', 'shopping'])
    ['wiki', 'wikis', 'shopping']

    We want 'wikis' -> 'wiki'. So:
    >>> lem2 = Lemmatizer(wnl=False).fit(labels)
    >>> lem2.transform(X, ['wiki', 'wikis', 'shopping'])
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
        self.wnl = False
        if wnl:
            try:
                from nltk.stem import WordNetLemmatizer
                self.wnl = WordNetLemmatizer()
            except ImportError:
                pass

        self.goodlist = frozenset(goodlist)
        self.categories = categories

    def fit(self, y):
        """'Fit' lemmatizer. Creates lemmatized classes_ list."""
        self.classes_ = sorted(set(self._transform(list(y) + self.categories)))
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
    
    Categorizes urls as one of a **predefined set**:
        blog | wiki | news | forum | classified | shopping | _UNDEF_ | _ERROR_
    Returns best guess and a dictionary of scores, which may be empty.
    See `classes_` for list of known classes.
    
    """
    
    classes_ = 'blog,wiki,news,forum,classified,shopping'.split(',')
    classes_.extend([UNDEF, ERROR])
    
    def __init__(self,
                 goldwords: dict=None,
                 offline=True,
                 θ=0.40,
                 n_jobs=cpu_count() - 1):
        """Set up the JPL Page Classifier.
        
        :param goldwords: dict {label -> "golden words" related to that category
            Default: use get_goldwords() to load them from category files
        :param offline: bool - True if HTML can be found in standard file loc'n
        :param θ: float - If no score > θ, returns UNDEF.
        :param n_jobs: int - Set ≤1 to disable parallel.
                
        """
        self.offline = offline
        self.θ = θ
        self.n_jobs = n_jobs
        if not goldwords:
            self.goldwords = get_goldwords(self.classes_, KEYWORD_DIR)
        else:
            self.goldwords = goldwords
        self.bleached = []
        self.errors = []
        self._estimator_type = "classifier"

    def fit(self, X, y=[]):
        """Not really fitting, just checks/updates self.classes_.
        
        :param X: list[str] - a list of URLs
        :param y: list[str] - a list of page types or classes
        
        """
        extras = [x for x in set(list(y)) if x not in self.classes_]
        if len(extras) > 0:
            s = "Found %d undeclared categories during 'fitting':\n\t->%s" % (len(extras), extras)
            logging.warning(s)
        #X, y = check_X_y(X, y)
        return self

    def predict(self, X):
        """Return the most likely class, for each x in X. Store probs in self.P."""
        self.P = self.predict_proba(X)
        label = np.vectorize(lambda x: self.classes_[x])
        return label(self.P.argmax(axis=1))

    def predict_proba(self, X, π=40):
        """For each x in X, provide vector of probs for each class.
            :param X: data, a sequence of URLs 
            :param π: int, threshold for parallelizing. Below this don't bother.
        
        URLs: JPL7 model assumes`get_html(url)` will retrieve HTML as required.
        
        Parallel logic modified from QingKaiKong. Also viewed pomegranate and scikit-issues.
            * http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html
            * https://github.com/jmschrei/pomegranate/blob/master/pomegranate/parallel.pyx
            * https://github.com/scikit-learn/scikit-learn/issues/7448
        
        """
        # check_is_fitted(self, ['X_', 'y_'])
        # X = check_array(X)
        if type(X) is str:
            raise(AttributeError, "predict_proba: X must be array-like, not string!")

        n, n_jobs = len(X), self.n_jobs
        if n < π:
            n_jobs = 1 # Not worth the overhead to parallelize
            batches = X
        else:
            starts = [i * n // n_jobs for i in range(n_jobs)]
            ends = starts[1:] + [n]
            batches = [X[start:end] for start, end in zip(starts, ends)]

        t0 = arrow.now()
        if n_jobs > 1:
            score = delayed(batch_score_urls)
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(score(batch, self) for batch in batches)
        else:
            results = (batch_score_urls(batch, self) for batch in batches)
        results = np.concatenate([np.array([row for row in batch]) for batch in results])
        dt = arrow.now() - t0

        logging.info('TIMING: n_jobs = %d, t = %s, dt = **%3.3fs**' %
                     (n_jobs, t0.format('HH:mm:ss'), dt.total_seconds()))
        return results


def batch_score_urls(batch: 'Series or Sequence we can promote to Series',
                     object: 'an instance of JPLPageClass'):
    """Batch wrapper makes it much easier to parallelize.
    Expects DataFrame or Series. Else tries to coax to DataFrame and continue. """
    try:
        return batch.apply(score_url, args=(object,))
    except AttributeError:
        logging.info("Trying to convert to Series: %s" % batch)
        return Series(batch).apply(score_url, args=(object,))


def score_url(url: 'The URL ', object: 'an instance of JPLPageClass'):
    """Score the URL using JPL cascade. Only parse HTML if URL inconclusive.
    
    Moved outside the class so joblib can pickle. 
    Now that we wrap with batch_score_urls, possibly could move this
    back into the JPL class.

    :param url: The url to score 
    :param object: The JPL classifier object.
    :return: score vector (numpy array)
    """
    # TODO: Move definitions back to predict_proba, and pass in, to avoid 'self' lookups.
    logging.debug('URL: %s' % url[:MAX_URL_LEN])
    𝜃 = object.θ
    scores = np.ones(len(object.classes_)) * .1
    idx = dict([(key,i) for i, key in enumerate(object.classes_)])
    def tally(key, val):
        scores[idx[key]] = val

    # 1. Check for blog goldwords in URL
    if url_has(url, object.goldwords['blog']):
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
    logging.info('score_url: URL = %s' % url)
    html = get_html(url, offline=object.offline)
    if html.startswith(HTTP_ERROR):
        #logging.warning('%s: %s ' % (HTTP_ERROR, clean_url(url)))
        object.errors.append(url)
    vals = get_cosines(html, object.goldwords)
    for key, val in vals.items():
        tally(key, val)
    if max(scores) > 𝜃:
        return scores / sum(scores)

    # Fallback
    tally(UNDEF, 1 - max(scores))
    return scores / sum(scores)


def evaluate(X, y, predicted, clf):
    """Evaluate the classifier based on its predictions."""
    classes_ = clf.classes_
    print(metrics.classification_report(y, predicted, labels=classes_))
    print("Confusion Matrix:")
    for row in zip(classes_,
                   metrics.confusion_matrix(y, predicted, labels=classes_)):
        print('%20s: %s' % (row[0],
                            ','.join(['%4d' % x for x in row[1]])))
    print("\n   µ Info: %4.2f" % metrics.adjusted_mutual_info_score(y, predicted))

    # Homebrew reporting
    model = clf.steps[-1][1]
    print('   Total #: %4d' % len(X))
    print('#Predicted: %4d' % len(predicted))
    print('  Accuracy: %4.2f' % np.mean(predicted == y))
    print("\n%d Errors -- HTML started with %s." % (len(model.errors), HTTP_ERROR))
    print("Either get new HTML or label these as category 'error'.")
    for row in model.errors:
        print('\t', row)
    # print(metrics.auc(labels, predicted))
    # print("ROC Curve:")
    # print(metrics.roc_curve(labels, predicted))

    # df, df_err, report = score_df(df, answers, probs)
    # df.to_csv(SCORE_FILE)  # , float_format='5.3f')
    # df.to_json(ERR_FILE)
    # print("\nURLs with Errors\n---------------\n", df_err)
    # print("Errors also saved to", ERR_FILE)
    # print(report)

#check_estimator(JPLPageClass)
if __name__ == "__main__":
    import pandas as pd

    #ERR_FILE = 'url_errs.json'
    URL_FILE = '../thh-classifiers/dirbot/full_urls.json'
    MAX_N = 500
    logging.info("Running [%s] as __main__.\n"
                 "-----------------------------------------------------\n"
                 "Training file: [%s].\n"
                 "OFFLINE = %s, MAX_N = %d\n"
                 % (__file__, URL_FILE, OFFLINE, MAX_N))
    df = pd.read_json(URL_FILE)
    df = df.sample(n=MAX_N, random_state=42)  # Subset for testing
    X = df.url
    y = Lemmatizer(wnl=True).fit_transform(df.pagetype)
    clf = make_jpl_clf(X, y, offline=OFFLINE)
    #probs = clf.predict_proba(df)
    predicted = clf.predict(X)
    evaluate(X, y, predicted, clf)

