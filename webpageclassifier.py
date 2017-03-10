# -*- coding: utf-8 -*-
# webpageclassifier.py

import collections
import itertools
import math
import re
import os.path
from time import sleep

import requests
from bs4 import BeautifulSoup, SoupStrainer

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

__author__ = 'Asitang Mishra jpl memex'

categories = 'blog, classified, forum, news, shopping, wiki, undecided'.split(', ')
PAGES_DIR = os.path.dirname(__file__) + '/Pages/'
KEYWORD_DIR = os.path.dirname(__file__) + '/Keywords/'
HTTP_ERROR = '_HTTP_ERROR_\n'

def read_golden(filepath):
    """Reads a golden file and creates canonical (lowercase) versions of each word.
    Returns a list
    """
    goldenlist = []
    with open(filepath, 'r', encoding='cp1252', errors='ignore') as f:
        goldenlist = [x.lower().strip() for x in f.readlines()]
    return goldenlist


def get_goldwords():
    gold_words = {}
    for name in ['blog', 'forum', 'news', 'shopping', 'classified']:
        gold_words[name] = read_golden(KEYWORD_DIR + name + '.txt')
    return gold_words


gold_words = get_goldwords()

# creates n grams for a string and outputs it as a list
def ngrams(input, n):
    """Creates n-grams for a string, returning a list.
     :param n: - n-gram length
     :returns: list
    """
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


# checks for the existence of a set of words (provided as a list) in the url
def word_in_url(url, wordlist):
    for word in wordlist:
        if word in url:
            return True
    return False

def flatten(l):
    """From Christian @ http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def extract_all_classnames(taglist, html_doc):
    """Extracts all `class` values `html_doc`, but only for tags in `taglist`.
    Ignores tags w/o class attribute - they don't affect cosine_sim anyway.
    Returns: flattened generator of class names appearing in tags.
    Note: returned generator may have "" entries, e.g. for <a class="" href=...>
    """
    # Note '_' in next line - soup trick to avoid the Python 'class' keyword.
    strainer = SoupStrainer(taglist, class_=True)
    soup = BeautifulSoup(html_doc, 'lxml', parse_only=strainer)
    return flatten((tag.attrs['class'] for tag in soup.find_all() if 'class' in tag.attrs))


def extract_all_fromtag(taglist, html_doc):
    """Extract all tags in taglist from html_doc. Return as list of Tag.
    Note some items will be long portions of the document!!
    """
    strainer = SoupStrainer(taglist)
    soup = BeautifulSoup(html_doc, 'lxml', parse_only=strainer)
    return soup.find_all()


# def numberoftags(taglist,html_doc):
# 	soup = BeautifulSoup(html_doc, 'lxml')
# 	count=0
# 	for tag in taglist:
# 		for classtags in soup.findall(tag):
# 				count+=1"""


def cosine_sim(words, goldwords):
    """Finds the normalized cosine overlap between two texts given as lists.
    """
    # TODO: Speed up the loops? If profile suggests this is taking any time.
    wordfreq = dict()
    goldwordfreq = dict()
    commonwords = []
    cosinesum = 0
    sumgoldwords = 0
    sumwords = 0

    for goldword in goldwords:
        if goldword in goldwordfreq.keys():
            goldwordfreq[goldword] = goldwordfreq[goldword] + 1
        else:
            goldwordfreq[goldword] = 1

    for word in words:
        if word in wordfreq.keys():
            wordfreq[word] = wordfreq[word] + 1
        else:
            wordfreq[word] = 1

    for word in goldwords:
        if word in wordfreq.keys():
            if word in goldwordfreq.keys():
                commonwords.append(word)
                cosinesum += goldwordfreq[word] * wordfreq[word]

    for word in goldwords:
        sumgoldwords += goldwordfreq[word] * goldwordfreq[word]

    for word in commonwords:
        sumwords += wordfreq[word] * wordfreq[word]

    # print(commonwords)

    sumwords = math.sqrt(sumwords)
    sumgoldwords = math.sqrt(sumgoldwords)
    if sumgoldwords != 0 and sumwords != 0:
        return cosinesum / (sumwords * sumgoldwords)
    return 0


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
        url_type = 'undecided'
    return url_type


def printlist(name, mylist, N=10, prefix='\t'):
    """Print first N items of list or generator, prefix & name"""
    try:
        print('{}{}: {}...'.format(prefix, name, mylist[:N]))
    except TypeError:
        ans = itertools.islice(mylist, N)
        print('{}{}: {}...'.format(prefix, name, ans))


def forum_score(html, forum_classnames):
    """Return cosine similarity between the forum_classnames and
    the 'class' attribute of certain tags.
    """
    tags = ['tr', 'td', 'table', 'div', 'p', 'article']
    classlist = extract_all_classnames(tags, html)
    #printlist('forum classlist:', classlist)
    # Keep only matches, and only in canonical form. So 'forum' not 'forums'.
    # TODO: doesn't this artificially inflate cosine_sim? By dropping non-matches?
    classlist = [j for i in classlist for j in forum_classnames if j in i]
    #printlist('canonical form :', classlist)

    return cosine_sim(classlist, forum_classnames)

def news_score(html, news_list):
    """Check if a news website: check the nav, header and footer data
    (all content, class and tags within), use similarity
    """
    tags = ['nav', 'header', 'footer']
    contents = extract_all_fromtag(tags, html)
    contents = (re.sub('[^A-Za-z0-9]+', ' ', x.text).strip() for x in contents)
    contents = ' '.join(contents).split(' ')
    #printlist('news contents:', contents)
    return cosine_sim(contents, news_list)

def read_url(url):
    """Fetch HTML from web, & convert to lowercase. If error, prepend with '_HTTP_ERROR_'.
    * Uses requests
    * Tries multiple user agents.
    * Logs errors if encountered.

     :param url: - the full URL. If blank, immediately returns error.
     :returns: - string, the HTML plus possible error prefix.

    """
    if url is None:
        return HTTP_ERROR + ": Empty URL.\n"

    # Some pages dislike custom agents. Define alternatives.
    alt_agents = [
        'Mozilla/5.0',
        'MEMEX_PageClass',
        'Gecko/1.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0',
        'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'
        'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
        ]

    for agent in alt_agents:
        r = requests.get(url, params={'User-Agent': agent})
        if r.status_code == requests.codes['ok']:
            return r.text.lower()
        wait = 1
        if r.status_code == requests.codes['too_many']:
            wait = int(r.headers['Retry-After'])
        print('*** Agent "%s" failed. --> Retrying <--' % agent[:20])
        sleep(wait) # Reduce chance of 429 error (Too many requests)

    # ERROR: Print some diagnostics and return error flag.
    print("\tERROR  :", r.status_code)
    print("\tCOOKIES:", [x for x in r.cookies])
    print("\tHISTORY:", r.history)
    print("\tHEADERS:", r.headers)
    print("\tRESPONSE:", r.text[:140].replace('\n','\n\t'), '...')
    return HTTP_ERROR + r.text.lower()


def get_html(url, filename, offline=True):
    """Get HTML from local file or the web. If web, write local file to filename."""
    if offline:
        print("\tOFFLINE mode: looking in %s." % PAGES_DIR)
        try:
            with open(PAGES_DIR + filename) as f:
                html = f.read()
            return html
        except OSError:
            print("Failed to find %s offline. Trying live URL." % filename)

    html = read_url(url)
    with open(filename, 'w') as f:
        f.write(html)
    return html


def get_cosines(text, gold, vals={}):
    """Calculate all cosine similarity scores.
    :param text: - str, the HTML or text to score
    :param gold: - list, the list of gold words or key words
    :param vals: - dict of scores by name, including: forum, news, classified, shopping
    :returns: - updated dict of scores, overwriting those 4 items
    
    """
    vals['forum'] = forum_score(text, gold['forum'])
    vals['news'] = news_score(text, gold['news'])
    text = re.sub(u'[^A-Za-z0-9]+', ' ', text)
    text_list = text.split(' ') + [' '.join(x) for x in ngrams(text, 2)]
    vals['classified'] = cosine_sim(text_list, gold['classified'])
    vals['shopping'] = cosine_sim(text_list, gold['shopping'])

    return vals


def categorize_url(url, goldwords, html=None, offline=False):
    """Categorizes urls as blog | wiki | news | forum | classified | shopping | undecided.
    Returns best guess and a dictionary of scores, which may be empty.
    """
    THRESH = 0.40
    scores = dict(((cat, 0) for cat in categories))
    if url is not None:
        # 1. Check for blog goldwords in URL
        if word_in_url(url, goldwords['blog']):
            scores['blog'] = .9
            return 'blog', scores

        # 2. Check for category name in URL
        name_type = name_in_url(url)
        if name_type != 'undecided':
            scores[name_type] = .9
            return name_type, scores

    # OK, we actually have to look at the page.
    if html is None:
        name = clean_url(url)
        url = expand_url(url)
        html = get_html(url, '{}.html'.format(name), offline=offline)
    if html.startswith('_HTTP_ERROR_'):
        return 'ERROR', scores
    scores = get_cosines(html, goldwords, scores)
    best = max(zip(scores.values(), scores.keys()))[1]
    if scores[best] > THRESH:
        return best, scores
    scores['undecided'] = 1 - scores[best]

    # 6. If still undecided, call hyperion grey classifier
    # if url_type=='undecided':
    #     fs = DumbCategorize(url)
    #     category=fs.categorize()
    #     url_type=category
    #     return url_type

    return 'undecided', scores

def expand_url(url):
    """Add http:// if not already there. Use before browsing."""
    if url.startswith('http'):
        return url
    else:
        return('http://' + url)


def clean_url(url, length=25):
    """Remove http[s]://; Replace / with |; Clip at _length_ chars."""
    if url.startswith('http'):
        start = url.index('//') + 2
        url = url[start:]
    return url[:length].replace('/', '|')

def print_weights(weights, prefix='\t[', suffix=']'):
    ans = []
    for key in ['forum', 'news', 'classified', 'shopping']:
        ans.append('%s: %4.2f' % (key[:2], weights[key]))
    print('{}{}{}'.format(prefix, ', '.join(ans), suffix))


def _accuracy(df, verbose=True):
    n_right = df['Correct?'].sum()
    acc = 1. * n_right / len(df)
    if verbose:
        print('\n*ACCURACY*: {}/{} = {:4.2f}'.format(n_right, len(df), acc))
    return acc


def score_df(df, answers, scores):
    """Compare df to answers and scores. Add answers & scores to df.
    Prints some scores along the way.
    :returns: pd.DataFrame, df with new columns

    """
    df['Test'] = answers
    df['Correct?'] = df['Test'] == df['Category']
    df = pd.concat([df, pd.DataFrame(scores)], axis=1)
    acc = _accuracy(df)
    return df


if __name__ == "__main__":
    import sys

    OFFLINE = True
    if OFFLINE:
        print("Running in OFFLINE mode.")
    import pandas as pd
    import __init__
    for key, val in gold_words.items():
        printlist(key, val)
    with open('urls.csv') as f:
        df = pd.read_csv(f, header=0, skipinitialspace=True)
    #df = df.iloc[:5]   # Subset for testing

    answers, scores = [], []
    for url in df['URL']:
        print('\n' + expand_url(url))
        cat, weights = categorize_url(url, gold_words, offline=OFFLINE)
        try:
            print_weights(weights)
        except KeyError:
            pass
        print('\t---> %s <--- ' % cat)
        answers.append(cat)
        scores.append(weights)

    df = score_df(df, answers, scores)
    df.to_csv('scores.csv')  # , float_format='5.3f')

