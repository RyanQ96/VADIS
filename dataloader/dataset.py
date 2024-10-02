import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast, PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class TrainingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract question and context from the dataset
        item = self.dataset[idx]
        if 'question' in item and 'context' in item:
            query = item['question']
            document = item['context']
        elif 'Question' in item and 'Context' in item:
            query = item['Question']
            document = item['Context']
        else:
            raise ValueError("Dataset format not recognized.")
 
        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Tokenize document
        doc_encoding = self.tokenizer(
            document,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'doc_input_ids': doc_encoding['input_ids'].squeeze(0),
            'doc_attention_mask': doc_encoding['attention_mask'].squeeze(0),
        }
        return item 
