# -*- coding: utf-8 -*-
import urltools
import json
import begin

"""Convert JSON list of URLs and pagetypes etc. into simple
 JSON_like list of sites to crawl like:
    {
        "plex": "https://forums.plex.tv/",
        "reddit": "http://www.reddit.com/",
        "reference": "http://www.reference.com/",
    }

"""
NAMES = {}  # Default dict of names we've seen so far


def get_sitename(url, hash=NAMES):
    """Extract a short(?) sitename from url. Ensure it is unique in "hash".
    First tries the site's domain. If that's in hash, add TLD. Then add
    integers until unique, insert {name: url} into hash, and return.

    NOTE: You must clear the hash if you want to start over!

    # Ignore folders & files
    >>> get_sitename("forums.xbox.com/en-US/home")
    'xbox'

    # Duplicate domain, though different TLD.
    >>> get_sitename("fake.xbox.org")
    'xbox.org'

    # Handle compound TLD "co.uk"
    >>> get_sitename("http://www.example.co.uk/abc")
    'example'

    # Repeat site - with proper TLD
    >>> get_sitename("www.example.co.uk")
    'example.co.uk'

    # Again -> get integer
    >>> get_sitename("www.example.co.uk/frog")
    'example.co.uk_02'

    # Again -> increment integer
    >>> get_sitename("www.example.co.uk/frog")
    'example.co.uk_03'

    # Send an empty hash - allows "example" again.
    >>> get_sitename("www.example.co.uk/frog", {})
    'example'


    """
    salt = 1
    ans = urltools.extract(url)
    name = ans.domain
    if name in hash:
        name += '.' + ans.tld
    while name in hash:
        if salt > 1:
            name = name[:-3]
        salt += 1
        name += "_{:02d}".format(salt)
    hash[name] = url
    return name


@begin.start(auto_convert=True)
def main(infile: 'Input JSON file', outfile

: 'Output JSON file', urlcol: 'Name of URL column' = 'url'):
"""Convert JSON records with named field 'url' into simple
 named JSON dict of sites to crawl like:
    {
        "plex": "https://forums.plex.tv/",
        "reddit": "http://www.reddit.com/",
        "reference": "http://www.reference.com/",
    }
If the extracted name already exists, add TLD, then add integers
until unique.

EXAMPLE: python crawllist.py 50urls.json crawlurls.json

"""
with open(infile) as f:
    items = json.load(f)
ans = {}
for item in items:
    url = item['url']
    name = get_sitename(url)
    ans[name] = url
with open(outfile, 'w') as f:
    json.dump(ans, f, sort_keys=True, indent=4)
