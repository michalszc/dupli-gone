import requests
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

DATASETS = [
    'tafenoquine',
    'uti',
    'diabetes',
    'copper',
    'blue-light',
]

BASE_URL = 'https://raw.githubusercontent.com/IEBH/dedupe-sweep/master/test/data/'

def load_xml_from_url(url: str) -> pd.DataFrame:
    '''
    Load XML data from a URL and convert it to a pandas DataFrame.
    '''
    r = requests.get(url)

    root = ET.fromstring(r.text)
    records = root.find('records') 

    def extract(child):
        parts = []
        for sub in child:
            if len(sub) != 0:
                parts.append(extract(sub))
                continue

            if sub.tag.lower() in ('_face','_font','_size'):
                continue

            if sub.text and sub.text.strip():
                parts.append(sub.text.strip())
        text = ','.join(parts)

        return text

    rows = []
    for rec in records.findall('record'):
        row = {}
        for child in rec:
            text = ''
            tag = child.tag
            if len(child) == 0:
                text = (child.text or '').strip()
            else:
                text = extract(child)

            row[tag] = text
        rows.append(row)

    return pd.DataFrame(rows)

def prepare_dataset() -> pd.DataFrame:
    '''
    Load and prepare the dataset for training and evaluation.
    '''
    dfs = [
        load_xml_from_url(f'{BASE_URL}{dataset}.xml')
        for dataset in tqdm(DATASETS, desc="Loading datasets")
    ]
    df = pd.concat(dfs, ignore_index=True)
    df['label'] = df['caption'].apply(lambda x: 1 if x == 'Duplicate' else 0)
    df.dropna(subset=['abstract'], inplace=True)

    return df
