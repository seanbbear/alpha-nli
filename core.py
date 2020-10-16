import jsonlines
import logging
from nlp import load_dataset
import numpy as np
import torch
from torch.utils.data import TensorDataset


def get_dataset(tokenizer, split):
    if split == 'train':
        path = './train-combine.jsonl'
    elif split == 'dev':
        path = './dev-combine.jsonl'

    with jsonlines.open(path) as f:
        data_len = len(list(f))
    
    with jsonlines.open(path) as f:
        input_ids = np.zeros(shape=(data_len,512))     
        token_type_ids = np.zeros(shape=(data_len,512))     
        attention_mask = np.zeros(shape=(data_len,512))     
        answer = []   
        
        index = 0
        
        # print(obj['label'])
        # print(len(list(f)))
        for obj in f:
            # seq class
            tensor_features = tokenizer(obj['obs1']+obj['obs2'], obj['hyp1']+obj['hyp2'], return_tensors='np',padding='max_length')
            
            # multi choice
            # context = obj['obs1'] + obj['obs2']
            # tensor_features = tokenizer( [context, context], [ obj['hyp1'], obj['hyp2'] ], return_tensors='np',padding='max_length')
            
            input_ids[index] = tensor_features['input_ids']
            token_type_ids[index] = tensor_features['token_type_ids']
            attention_mask[index] = tensor_features['attention_mask']
            answer.append(int(obj['label'])-1)
            # label只能是0跟1 
            
            index += 1

        input_ids = torch.LongTensor(input_ids)     
        token_type_ids = torch.LongTensor(token_type_ids)      
        attention_mask = torch.LongTensor(attention_mask)     
        answer = torch.LongTensor(answer) 
    return TensorDataset(input_ids, token_type_ids, attention_mask, answer)

def get_dataset_multi_choice(tokenizer, split):
    if split == 'train':
        path = './train-small.jsonl'
    elif split == 'dev':
        path = './dev-small.jsonl'

    with jsonlines.open(path) as f:
        data_len = len(list(f))
    
    with jsonlines.open(path) as f:
        input_ids = np.zeros(shape=(data_len,1024))     
        token_type_ids = np.zeros(shape=(data_len,1024))     
        attention_mask = np.zeros(shape=(data_len,1024))     
        answer = []   
        
        index = 0
        for obj in f:
            # seq class
            # tensor_features = tokenizer(obj['obs1']+obj['obs2'], obj['hyp1']+obj['hyp2'], return_tensors='np',padding='max_length')
            
            # multi choice
            context = obj['obs1'] + obj['obs2']
            tensor_features = tokenizer( [context, context], [ obj['hyp1'], obj['hyp2'] ], return_tensors='np',padding='max_length')
            
            input_ids[index] = tensor_features['input_ids'].reshape(1024)
            token_type_ids[index] = tensor_features['token_type_ids'].reshape(1024)
            attention_mask[index] = tensor_features['attention_mask'].reshape(1024)
            answer.append(int(obj['label'])-1)
            # label只能是0跟1 
            
            index += 1

        input_ids = torch.LongTensor(input_ids)     
        token_type_ids = torch.LongTensor(token_type_ids)      
        attention_mask = torch.LongTensor(attention_mask)     
        answer = torch.LongTensor(answer) 
    return TensorDataset(input_ids, token_type_ids, attention_mask, answer)


def compute_accuracy(y_pred, y_target):
    # 計算正確率
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100  

def model_setting(model_name):
    if model_name=='bert':
        from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig            
        config = BertConfig.from_pretrained("bert-base-uncased",num_labels = 2)              
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")              
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased") 
        return config, tokenizer, model
            
    
    elif model_name=='albert-seq-classify':
        from transformers import AutoTokenizer, AlbertForSequenceClassification, AlbertConfig   
        config = AlbertConfig.from_pretrained("albert-base-v2", classifier_dropout_prob = 0.2)     
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")     
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2")
        return config, tokenizer, model
    
    elif model_name=='albert-multi-choice':
        from transformers import AutoTokenizer, AlbertForMultipleChoice, AlbertConfig   
        config = AlbertConfig.from_pretrained("albert-base-v2")     
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")     
        model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")
        return config, tokenizer, model


if __name__ == "__main__":
    config, tokenizer, model = model_setting('albert')
    # train_dataset = get_dataset(name="boolq", tokenizer=tokenizer, split='train')
    # print(type(train_dataset))
    get_dataset(tokenizer)
    
    


































