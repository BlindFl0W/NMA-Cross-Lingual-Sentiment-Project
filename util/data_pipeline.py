import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict, Dataset
from typing import Tuple
import spacy
from spacy.cli import download as spacy_download
import re

def get_data_loaders(tokenizer, presentage: float, batch_size: int, preprocessing: bool = False) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    dataset = download_data()
    english_train_dataset, english_validation_dataset, target_validation_dataset, target_test_dataset = get_train_validation_test_split(dataset, presentage)
    english_train_df = get_dataframes(english_train_dataset, preprocessing)
    english_validation_df = get_dataframes(english_validation_dataset, preprocessing)
    target_validation_df = get_dataframes(target_validation_dataset, preprocessing)
    target_test_df = get_dataframes(target_test_dataset, preprocessing)

    # Tokenize the 'text' column
    # Using padding='max_length' and truncation=True to handle variable lengths
    english_train_encodings = tokenizer(english_train_df['text'].tolist(), truncation=True, padding='max_length', max_length=512)
    english_validation_encodings = tokenizer(english_validation_df['text'].tolist(), truncation=True, padding='max_length', max_length=512)
    target_validation_encodings = tokenizer(target_validation_df['text'].tolist(), truncation=True, padding='max_length', max_length=512)
    target_test_encodings = tokenizer(target_test_df['text'].tolist(), truncation=True, padding='max_length', max_length=512)

    # Create TensorDatasets
    english_train_labels = torch.tensor(english_train_df['label'].tolist())
    english_validation_labels = torch.tensor(english_validation_df['label'].tolist())
    target_validation_labels = torch.tensor(target_validation_df['label'].tolist())
    target_test_labels = torch.tensor(target_test_df['label'].tolist())

    english_train_dataset = TensorDataset(
        torch.tensor(english_train_encodings['input_ids']),
        torch.tensor(english_train_encodings['attention_mask']),
        english_train_labels
    )

    english_validation_dataset = TensorDataset(
        torch.tensor(english_validation_encodings['input_ids']),
        torch.tensor(english_validation_encodings['attention_mask']),
        english_validation_labels
    )

    target_validation_dataset = TensorDataset(
        torch.tensor(target_validation_encodings['input_ids']),
        torch.tensor(target_validation_encodings['attention_mask']),
        target_validation_labels
    )

    target_test_dataset = TensorDataset(
        torch.tensor(target_test_encodings['input_ids']),
        torch.tensor(target_test_encodings['attention_mask']),
        target_test_labels
    )

    english_train_loader = DataLoader(english_train_dataset, batch_size=batch_size, shuffle=True)
    english_validation_loader = DataLoader(english_validation_dataset, batch_size=batch_size, shuffle=False)
    target_validation_loader = DataLoader(target_validation_dataset, batch_size=batch_size, shuffle=False)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

    return english_train_loader, english_validation_loader, target_validation_loader, target_test_loader

def preprocess_data(text, language_model=None):
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    text = text.encode('utf-8').decode('utf-8-sig').lower()
    text = re.sub(r'(RT @\w+|http\S+|www\S+|@\S+|&[a-z]+;|\n|\r)', '', text)
    if not language_model:
        return text
    try:
        nlp = spacy.load(language_model)
    except OSError:
        print(f"Model '{language_model}' not found. Downloading now…")
        spacy_download(language_model)
        nlp = spacy.load(language_model)
    doc = nlp(text)
    words = [
        token.lemma_
        for token in doc
        if not token.is_punct
        and not token.is_digit
        and not token.is_stop
        and not token.is_space
        and token.is_alpha
    ]
    return ' '.join(words)

def download_data() -> DatasetDict:
    dataset = load_dataset("clapAI/MultiLingualSentiment")
    return dataset

def get_english_dataset(dataset: DatasetDict) -> DatasetDict:
    english_dataset = dataset.filter(lambda example: example['language'] == 'en')
    return english_dataset

def get_forgin_latin_dataset(dataset: DatasetDict) -> DatasetDict:
    target_dataset = dataset.filter(lambda example: example['language'] == 'es' or example['language'] == 'fr' or example['language'] == 'it')
    return target_dataset

def get_subset_dataset(dataset: DatasetDict, percentage: float) -> DatasetDict:
    dataset = dataset.shuffle(seed=1).select(range(int(dataset.num_rows * percentage)))
    return dataset

def get_train_validation_test_split(dataset: DatasetDict, percentage: float) -> DatasetDict:
    english_dataset = get_english_dataset(dataset)
    target_dataset = get_forgin_latin_dataset(dataset)

    english_train_dataset = get_subset_dataset(english_dataset['train'], percentage)
    english_validation_dataset = get_subset_dataset(english_dataset['validation'], percentage)
    target_validation_dataset = get_subset_dataset(target_dataset['validation'], percentage)
    target_test_dataset = get_subset_dataset(target_dataset['test'], percentage)

    return english_train_dataset, english_validation_dataset, target_validation_dataset, target_test_dataset

def get_dataframes(dataset: DatasetDict, language_model="en_core_web_sm", preprocessing: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(dataset)
    df.drop(columns=['source', 'domain', 'language'], inplace=True)
    df['label'] = df['label'].map({'positive': 0, 'neutral': 1, 'negative': 2})
    if preprocessing:
        df['text'] = df['text'].apply(lambda x: preprocess_data(x, language_model))
    return df
