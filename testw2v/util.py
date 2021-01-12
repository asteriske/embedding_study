import datetime
import json
import os
import subprocess
from typing import Any, Dict
import urllib.request
import zipfile

def git_hash() -> str:

    process = subprocess.Popen(['git', 'log', '--pretty=oneline','-n','1'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()

    return git_head_hash.decode('utf-8').replace(' ','_')


def write_metrics(path: str, metrics: Dict[str,Any]) -> None:

    date_str = (str(datetime.datetime.now().isoformat())
        .replace('.','')
        .replace(':','')
        .replace('-','')
    )
    
    metrics_path = os.path.join(path, git_hash())

    if not os.path.isfile(metrics_path):
        try:
            os.makedirs(metrics_path)
        except FileExistsError:
            pass

    for k in metrics.keys():
        file_path = os.path.join(metrics_path, f"{k}_{date_str}.json")

        with open(file_path,'w') as f:
            json.dump(metrics[k],f, indent=2)


def simple_download(uri, destination):
    print(f"Downloading {uri}...")
    urllib.request.urlretrieve(uri, destination)


def prep_wikitext():
    simple_download('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip','wikitext-103-v1.zip')
    
    with zipfile.ZipFile('wikitext-103-v1.zip', 'r') as zip_obj:
        zip_obj.extractall()

        with open('wikitext-103/wiki.train.tokens','r') as source_file:

            for num_line, _ in enumerate(source_file.readlines()):
                pass

        with open('wikitext-103/wiki.train.tokens','r') as source_file:
            with open('wiki_10pct','w') as dest_file:
                for i, line in enumerate(source_file.readlines()):
                    dest_file.write(line)
                    if i > round(num_line/10.0):
                        break