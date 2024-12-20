# prepare_evaluation_data.py
import os
import requests
import zipfile
import tarfile
import shutil
from huggingface_hub import hf_hub_download

def download_and_extract(url, extract_path):
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
    else:
        print(f"{filename} already exists.")

    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif filename.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        print(f"Cannot extract {filename}.")

def prepare_ptb():
    url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
    os.makedirs('data/ptb', exist_ok=True)
    for split in ['train', 'valid', 'test']:
        split_url = url.replace('train', split)
        r = requests.get(split_url)
        with open(f'data/ptb/{split}.txt', 'w') as f:
            f.write(r.text)
    print("PTB dataset prepared.")

def prepare_wikitext2():
    import os
    from huggingface_hub import hf_hub_download

    repo_id = "Salesforce/wikitext"
    files = [
        "wikitext-2-v1/train-00000-of-00001.parquet",
        "wikitext-2-v1/validation-00000-of-00001.parquet",
        "wikitext-2-v1/test-00000-of-00001.parquet"
    ]
    extract_path = 'data/'
    os.makedirs(extract_path, exist_ok=True)

    print("Downloading WikiText-2 dataset from Hugging Face...")
    for file_path in files:
        local_path = os.path.join(extract_path, os.path.basename(file_path))
        if not os.path.exists(local_path):
            hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=extract_path, repo_type="dataset")
            print(f"Downloaded {os.path.basename(file_path)} to {extract_path}.")
        else:
            print(f"{os.path.basename(file_path)} already exists in {extract_path}.")
    print("WikiText-2 dataset preparation complete.")

def prepare_wikitext103():
    import os
    from huggingface_hub import hf_hub_download

    repo_id = "Salesforce/wikitext"
    files = [
        "wikitext-103-v1/train-00000-of-00002.parquet",
        "wikitext-103-v1/train-00001-of-00002.parquet",
        "wikitext-103-v1/validation-00000-of-00001.parquet",
        "wikitext-103-v1/test-00000-of-00001.parquet"
    ]
    extract_path = 'data/'
    os.makedirs(extract_path, exist_ok=True)

    print("Downloading WikiText-103 dataset from Hugging Face...")
    for file_path in files:
        local_path = os.path.join(extract_path, os.path.basename(file_path))
        if not os.path.exists(local_path):
            hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=extract_path, repo_type="dataset")
            print(f"Downloaded {os.path.basename(file_path)} to {extract_path}.")
        else:
            print(f"{os.path.basename(file_path)} already exists in {extract_path}.")
    print("WikiText-103 dataset preparation complete.")

def prepare_lambada():
    url = 'https://raw.githubusercontent.com/cybertronai/bflm/refs/heads/master/lambada_test.jsonl'
    os.makedirs('data/lambada', exist_ok=True)
    r = requests.get(url)
    with open('data/lambada/lambada_test.jsonl', 'wb') as f:
        f.write(r.content)
    print("LAMBADA dataset prepared.")

if __name__ == '__main__':
    prepare_ptb()
    prepare_wikitext2()
    prepare_wikitext103()
    prepare_lambada()
