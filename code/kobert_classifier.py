import torch
from torch import nn
import gluonnlp as nlp
from torch.utils.data import TensorDataset, Dataset
import pandas as pd
import numpy as np


class BERTDataset(Dataset):
    def __init__(self, dataset, idx_idx, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        
        self.indices = [np.int32(i[idx_idx]) for i in dataset]
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
    
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ) + (self.indices[i], ))
    
    def __len__(self):
        return (len(self.labels))

class BERTDataset_Ops(Dataset):
    def __init__(self, dataset, idx_idx, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        
        self.indices = [np.int32(i[idx_idx]) for i in dataset]
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
    
    def __getitem__(self, i):
        return (self.sentences[i] + (self.indices[i], ))
    
    def __len__(self):
        return (len(self.labels))


###################################
# KoBERT Classifier Classes:
# model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
# model = BERTClassifier_Softmax(bertmodel, dr_rate=0.5).to(device)
# model = BERTClassifier_HL(bertmodel, dr_rate=0.5).to(device)
# model = BERTClassifier_HLHL(bertmodel, dr_rate=0.5).to(device)
# model = BERTClassifier_HLHL_Softmax(bertmodel, dr_rate=0.5).to(device)
# reference: https://github.com/SKTBrain/KoBERT


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None
                ):
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_maks = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long()
                              attention_mask=attention_mask.float().to(token_ids.device)
                             )
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
        

class BERTClassifier_Softmax(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None
                ):
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_maks = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long()
                              attention_mask=attention_mask.float().to(token_ids.device)
                             )
        if self.dr_rate:
            out = self.dropout(pooler)
        out = self.classifier(out)
        return self.softmax(out)
    
    
class BERTClassifier_HL(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None
                ):
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, hidden_size // 2)
        self.classifier2 = nn.Linear(hidden_size // 2, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_maks = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long()
                              attention_mask=attention_mask.float().to(token_ids.device)
                             )
        if self.dr_rate:
            out = self.dropout(pooler)
        out = self.classifier(out)
        return self.classifier2(out)
    
    
class BERTClassifier_HLHL(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None
                ):
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_maks = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long()
                              attention_mask=attention_mask.float().to(token_ids.device)
                             )
        if self.dr_rate:
            out = self.dropout(pooler)
        out = self.classifier(out)
        return self.classifier2(out)    

    
class BERTClassifier_HLHL_Softmax(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None
                ):
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_maks = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long()
                              attention_mask=attention_mask.float().to(token_ids.device)
                             )
        if self.dr_rate:
            out = self.dropout(pooler)
        out = self.classifier(out)
        out = self.classifier2(out)
        return self.softmax(out)

