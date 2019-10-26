from .config import args
from transformers.optimization import AdamW, WarmupLinearSchedule


def get_optimizer(model, lr):       

    # Prepare optimiser and schedule 
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    return AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    
def get_scheduler(optimizer):
    return WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=args['total_steps'])


