# -*- coding: utf-8 -*-
# Needs Python3 to use the awesome "begins" module for simpler command-line parsing.
# Begins lets us avoid the if __name__ == blah blah.
import pandas as pd
import begin  # For Python < 3.3, import funcsigs or remove annotations.


def clean_json(jfile: 'JSON file from df.to_json()'

):
"""Clean up pandas df.to_json() file to be compliant and readable."""
with open(jfile) as f:
    urlstr = ','.join(f.readlines())
urlstr = urlstr.replace('{', '{\n    ').replace('}', '\n}').replace(',"', ',\n    "')
with open(jfile, 'w') as f:
    f.write('[' + urlstr + ']\n')


@begin.start(auto_convert=True)
def main(infile: 'Input CSV file', outfile

: 'Output JSON file', columns: 'list[str]' = None, crawllist: 'Make crawl list?' = False):
"""Convert CSV file to JSON file, optionally renaming the columns

EXAMPLE: python csv2json.py --columns pagetype,url

"""
df = pd.read_csv(infile, skipinitialspace=True)
if columns:
    df.columns = columns.split(',')
df.to_json(outfile, orient='records', lines=True)
clean_json(outfile)
