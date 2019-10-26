import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(model, test_data, test_dataloader, labels_list):
    
    # Hold input data for returning it 
    input_data = [{ 'id': input_test.guid, 'content': input_test.text } for input_test in test_data]
    
    all_logits = None
    model.eval()
    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids = batch
        # 将运算数据迁移到 gpu 上
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        if all_logits is None:
            all_logits = logits.detach().cpu()
        else:
            all_logits = torch.cat((all_logits, logits.detach().cpu()), 1)
        
    # pred_labels : [num_tasks, num_test]
    pred_labels = torch.argmax(all_logits, dim=2)
    
    # 因为预处理将标签 +2，所以这里再减去2
    pred_labels -= 2
    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(pred_labels.T.numpy(), columns=labels_list), left_index=True, right_index=True)

