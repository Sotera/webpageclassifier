# -*- coding: utf-8 -*-

import itertools
import requests
import logging
from bs4 import BeautifulSoup, SoupStrainer
from time import sleep

"""
Utils for webpageclassifier.

"""

def _accuracy(df, colname):
    """Calculate simple accuracy by summing column 'colname'. Return (n_right, N, acc)."""
    n_right = df[colname].sum()
    acc = 1. * n_right / len(df)
    return n_right, len(df), acc


def clean_url(url, length=25):
    """Remove http[s]://; Replace / with |; Clip at _length_ chars."""
    if url.startswith('http'):
        start = url.index('//') + 2
        url = url[start:]
    return url[:length].replace('/', '|')


def expand_url(url):
    """Add http:// if not already there. Use before browsing."""
    if url.startswith('http'):
        return url
    else:
        return('http://' + url)


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


def flatten(l):
    """Flattens an irregular list. Python 3.5+.
    http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_goldwords(names, folder):
    """Call read_golden foreach name in _names_, using _folder_."""
    gold_words = {}
    for name in names:
        gold_words[name] = read_golden(folder + name + '.txt')
    return gold_words


def get_html(url, filename=None, offline=True):
    """Get HTML from local file or the web. If web, write local file to filename."""
    name, url = clean_url(url), expand_url(url)
    if filename is None:
        filename = '{}.html'.format(name)
    if offline:
        logging.info("\tOFFLINE mode: looking in %s." % PAGES_DIR)
        try:
            with open(PAGES_DIR + filename) as f:
                html = f.read()
            return html
        except OSError:
            logging.info("Failed to find %s offline. Trying live URL." % filename)

    html = read_url(url)
    with open(PAGES_DIR + filename, 'w') as f:
        f.write(html)
    return html


def normalize(hash):
    """Return normalized copy of _hash_, dividing each val by sum(vals).
    
    :param hash: a key->number dict
    :return: dict
    
    >>> [(k,v) for k,v in sorted(normalize({1:3, 2:3, 3:6}).items())]
    [(1, 0.25), (2, 0.25), (3, 0.5)]
    
    """
    total = sum(hash.values())
    ans = {}
    for key, val in hash.items():
        ans[key] = val / total
    return ans


def print_weights(weights, prefix='\t[', suffix=']'):
    ans = []
    for key in ['forum', 'news', 'classified', 'shopping']:
        ans.append('%s: %4.2f' % (key[:2], weights[key]))
    print('{}{}{}'.format(prefix, ', '.join(ans), suffix))


def prettylist(name, mylist, N=10, prefix='\t'):
    """Return formatted str: first N items of list or generator, prefix & name"""
    try:
        return('{}{}: {}...'.format(prefix, name, mylist[:N]))
    except TypeError:
        ans = itertools.islice(mylist, N)
        return('{}{}: {}...'.format(prefix, name, ans))


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
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0',
        'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'
        'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
        ]

    for agent in alt_agents:
        try:
            r = requests.get(url, params={'User-Agent': agent})
        except requests.exceptions.RequestException as e:
            logging.info("\tConnection Error:", e)
            break
        if r.status_code == requests.codes['ok']:
            return r.text.lower()
        wait = 1
        if r.status_code == requests.codes['too_many']:
            wait = int(r.headers['Retry-After'])
        logging.warning('Agent "%s" failed. --> Retrying <--' % agent[:20])
        sleep(wait) # Reduce chance of 429 error (Too many requests)

    # ERROR: Print some diagnostics and return error flag.
    try:
        logging.warning("\tERROR   :", r.status_code)
        logging.warning("\tCOOKIES :", [x for x in r.cookies])
        logging.warning("\tHISTORY :", r.history)
        logging.warning("\tHEADERS :", r.headers)
        logging.warning("\tRESPONSE:", r.text[:200].replace('\n', '\n\t'), '...')
    except UnboundLocalError:
        logging.error("\tERROR   : r was undefined - no further information available")
        return HTTP_ERROR + "Connection Error: no response item"
    return HTTP_ERROR + r.text.lower()


def read_golden(filepath):
    """Reads a golden file and creates canonical (lowercase) versions of each word.
    Returns a list
    """
    goldenlist = []
    try:
        with open(filepath, 'r', encoding='cp1252', errors='ignore') as f:
            goldenlist = [x.lower().strip() for x in f.readlines()]
    except FileNotFoundError:
        logging.warning('Not found: %s. Making blank goldwords list.' % filepath)
    return goldenlist


def score_df(df, answers, scores, colname='pagetype', verbose=False):
    """Compare df to answers and scores. Add answers & scores to df.
    Prints some scores along the way.
    :param df: The dataframe with URLs and answers
    :param answers: List with predicted categories
    :param scores: List of dicts with category scores
    :param colname: String, name of category column
    :param verbose: Boolean
    :returns: df2, df_err, report : (df with valid rows & score columns,
                                    df of error urls,
                                    string with ERROR & ACC counts)

    """
    report = []
    df['Best'] = answers
    scores = pd.DataFrame(scores, index=df.index)
    df = pd.concat([df, scores], axis=1)
    df_errs = df['Best'] == 'ERROR'
    df2 = df[~(df_errs)]
    # Disable chained-assignments warnings for the next 3 lines
    # I'm pretty sure I'm not creating a copy of df2 here.
    pd.set_option('mode.chained_assignment', None)
    df2['Plural'] = df2['Best'].map(lambda x: x + 's')
    df2['Correct?'] = (df2['Best'] == df2[colname]) | (df2['Plural'] == df2[colname])
    pd.set_option('mode.chained_assignment', 'warn')

    if verbose:
        report.extend(["df2", "----", df2.__repr__()])
    n_df = len(df)
    n_right, n_ok, acc = _accuracy(df2, colname='Correct?')
    n_err = n_df - n_ok
    report.append("  *ERRORS*: {}/{} = {:4.2f}".format(n_err, n_df, n_err / n_df))
    report.append("*ACCURACY*: {}/{} = {:4.2f}".format(n_right, n_ok, acc))
    report = '\n'.join(report)
    return df2, df[df_errs].filter(['url', 'pagetype', 'Best']), report
