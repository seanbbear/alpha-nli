from core import get_dataset, compute_accuracy, model_setting, get_dataset_multi_choice
import torch

from torch.utils.data import DataLoader
from transformers import AdamW
import os
import wandb
from tqdm import tqdm

# seq classify
# if __name__ == "__main__": 
#     # wandb專案名稱
#     wandb.init(project="alpha-nli-classification")

#     config, tokenizer, model = model_setting('albert-multi-choice')
   
#     wandb.watch(model)

#     # setting device    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
#     # 多GPU
#     # print(torch.cuda.device_count())
#     if  torch.cuda.device_count()>1:         
#         model = torch.nn.DataParallel(model,device_ids=[0,1])
    
#     print("using device",device)
#     model.to(device)
    
#     # print(model.train())
   
#     train_dataset = get_dataset(tokenizer=tokenizer, split='train')
#     test_dataset = get_dataset(tokenizer=tokenizer, split='dev')

#     train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
#         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.1}
#         ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6, eps=1e-8)

#     model.zero_grad()
#     for epoch in tqdm(range(15)):
#         train_loss = 0.0
#         train_acc = 0.0
#         model.train()
#         for batch_index, batch_dict in enumerate(train_dataloader):
#             batch_dict = tuple(t.to(device) for t in batch_dict)
#             outputs = model(
#                 batch_dict[0],
#                 labels = batch_dict[3]
#             )
#             loss, logits = outputs[:2]
#             if (device=='cuda' and device,torch.cuda.device_count()>1):
#                 loss = loss.mean()

#             loss.backward()
#             optimizer.step()
#             model.zero_grad()

#             # 計算loss
#             loss_t = loss.item()
#             train_loss += (loss_t - train_loss)/(batch_index + 1)

#             # 計算accuracy
#             acc_t = compute_accuracy(logits, batch_dict[3])
#             train_acc += (acc_t - train_acc) / (batch_index + 1)

#             # log
            
#         print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, train_loss, train_acc))
#         wandb.log({"Train Acc":train_acc, "Train Loss":train_loss})
        
#         test_loss = 0.0
#         test_acc = 0.0
#         model.eval()
#         for batch_index, batch_dict in enumerate(test_dataloader):
#             batch_dict = tuple(t.to(device) for t in batch_dict)
#             outputs = model(
#                 batch_dict[0],
#                 labels = batch_dict[3]
#             )
#             loss,logits = outputs[:2]
#             if (device=='cuda' and device,torch.cuda.device_count()>1):                 
#                 loss = loss.mean()
            
            
#             # 計算loss
#             loss_t = loss.item()
#             test_loss += (loss_t - test_loss) / (batch_index + 1)

#             # 計算accuracy
#             acc_t = compute_accuracy(logits, batch_dict[3])
#             test_acc += (acc_t - test_acc) / (batch_index + 1)

#             # log
#         print("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f"%(epoch+1, batch_index+1, test_loss, test_acc))
#         wandb.log({"Test Acc":test_acc, "Test Loss":test_loss})
        
#         torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
#     model_to_save = model.module if hasattr(model, 'module') else model
#     model_to_save.save_pretrained('trained_model')


# multiple choice
if __name__ == "__main__": 
    # wandb專案名稱
    # wandb.init(project="alpha-nli-classification")

    config, tokenizer, model = model_setting('albert-multi-choice')
   
    # wandb.watch(model)

    # setting device    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 多GPU
    # print(torch.cuda.device_count())
    if  torch.cuda.device_count()>1:         
        model = torch.nn.DataParallel(model,device_ids=[0,1])
    
    print("using device",device)
    model.to(device)
    
    # print(model.train())
   
    train_dataset = get_dataset_multi_choice(tokenizer=tokenizer, split='train')
    test_dataset = get_dataset_multi_choice(tokenizer=tokenizer, split='dev')

    batch = 20
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.1}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6, eps=1e-8)

    model.zero_grad()
    for epoch in tqdm(range(15)):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for batch_index, batch_dict in enumerate(train_dataloader):
            batch_dict = tuple(t.to(device) for t in batch_dict)
            # 
            num_choices=batch_dict[0].shape[1]
            print(num_choices)
            # 
            input_ids = batch_dict[0].reshape(batch, 2, 512)
            attention_mask = batch_dict[1].reshape(batch, 2, 512)
            token_type_ids = batch_dict[2].reshape(batch, 2, 512)
            # print(token_type_ids.shape)

            # input_ids = input_ids.view(-1, input_ids.size(-1))
            # attention_mask = attention_mask.view(-1, attention_mask.size(-1)) 
            # token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            # print(token_type_ids.shape)
            outputs = model(
                input_ids,
                attention_mask,
                token_type_ids,
                labels=batch_dict[3]
            )
            loss, logits = outputs[:2]
            
            if (device=='cuda' and device,torch.cuda.device_count()>1):
                loss = loss.mean()
            print(loss)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            # 計算loss
            loss_t = loss.item()
            train_loss += (loss_t - train_loss)/(batch_index + 1)

            # 計算accuracy
            acc_t = compute_accuracy(logits, batch_dict[3])
            train_acc += (acc_t - train_acc) / (batch_index + 1)

            # log
            
        print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, train_loss, train_acc))
        # wandb.log({"Train Acc":train_acc, "Train Loss":train_loss})
        
        test_loss = 0.0
        test_acc = 0.0
        model.eval()
        for batch_index, batch_dict in enumerate(test_dataloader):
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                labels = batch_dict[3]
            )
            loss,logits = outputs[:2]
            if (device=='cuda' and device,torch.cuda.device_count()>1):                 
                loss = loss.mean()
            
            
            # 計算loss
            loss_t = loss.item()
            test_loss += (loss_t - test_loss) / (batch_index + 1)

            # 計算accuracy
            acc_t = compute_accuracy(logits, batch_dict[3])
            test_acc += (acc_t - test_acc) / (batch_index + 1)

            # log
        print("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f"%(epoch+1, batch_index+1, test_loss, test_acc))
        # wandb.log({"Test Acc":test_acc, "Test Loss":test_loss})
        
        # torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('trained_model')