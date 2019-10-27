import torch
import os
import numpy as np
from .config import args
from sklearn.metrics import f1_score
from tqdm import tqdm_notebook as tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(num_epochs, train_dataloader, eval_dataloader, model, optimizer, scheduler):

    model.train()
    for i_ in tqdm(range(int(num_epochs)), desc="Epoch"):

        train_loss = 0
        num_train, train_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # 将运算数据迁移到 gpu 上
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)

            loss.backward()

            train_loss += loss.item()
            num_train += input_ids.size(0)
            train_steps += 1
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 300 == 0:
                print('train_loss:', loss.item() / args['batch_size'])
                print('already computed :{}%', (step / args['total_steps'])*100)

        print('Train loss after epoc {}'.format(train_loss / train_steps / args['batch_size']))
        print('Eval after epoc {}'.format(i_+1))

        
        # 因为要运行很久，所以每个epoch 保存一次模型
        path = './directory/to/save/'
        if not os.path.exists(path):
            os.makedirs(path)
        model.save_pretrained(path)  
        
        eval(model, eval_dataloader)

def eval(model, eval_dataloader):
    
    all_logits = None
    all_labels = None
    
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    eval_steps, num_eval = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        # 将运算数据迁移到 gpu 上
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

            
        # 将各个batch 的 logits 和 labels 拼接在一起，用于 f1_score 计算
        # logits 维度 [num_tasks, batch, 4], label_ids ： [ batch, num_tasks]
        # all_logits :[num_tasks, num_eval, 4], all_label_ids : [num_eval, num_tasks]
        
        if all_logits is None:
            all_logits = logits.detach().cpu()
        else:
            all_logits = torch.cat((all_logits, logits.detach().cpu()), 1)
            
        if all_labels is None:
            all_labels = label_ids.detach().cpu()
        else:    
            all_labels = torch.cat((all_labels, label_ids.detach().cpu()), 0)
        
        eval_loss += tmp_eval_loss.item()

        num_eval += input_ids.size(0)
        eval_steps += 1

    eval_loss = eval_loss / eval_steps
    
    # Compute f1_scores
    f1_scores_list = []
    # pred_labels : [num_tasks, num_eval]
    pred_labels = torch.argmax(all_logits, dim=2)
    for i in range(args['num_tasks']):
        f1_scores_list.append(f1_score(all_labels[:,i].numpy(), pred_labels[i].numpy(), average='macro'))
        
    f1_scores  = np.mean(f1_scores_list)
    
    print('Eval loss after epoc {}'.format(eval_loss / args['batch_size']))
    print('f1_score after epoc {}'.format(f1_scores))


