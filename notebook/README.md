### 关于 MyBertForSequenceClassification 类的说明

因为 transformers 中的 BertForSequenceClassification 类只适合处理二分类问题，所以依照 BertForSequenceClassification  进行了修改，BertForSequenceClassification 所在的模块是 transformers/modeling_bert.py, 在 BertForSequenceClassification 类是下方添加以下代码：
```python
class MyBertForSequenceClassification(BertPreTrainedModel):

    """
    # 有6大类，下面共有20个标签，不考虑大类，直接考虑标签，每个标签有4个类别，可以将
    # 任务分为多任务分类问题，每一个任务都是一个4分类问题。
    # 参见 https://github.com/brightmart/sentiment_analysis_fine_grain
    """

    num_labels = 4
    num_tasks = 20

    def __init__(self, config):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = MyBertForSequenceClassification.num_labels
        self.num_tasks = MyBertForSequenceClassification.num_tasks

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建20个分类任务，每个任务共享输入： BertModel 的输出最后一层的 [CLS] 的 pooler_output
        # 但是源程序也说了，使用 [cls] 的 pooler_output is usually *not* a good summary
        # of the semantic content of the input, you're often better with averaging or pooling
        # the sequence of hidden-states for the whole input sequence.
        # module_list = []
        # for _ in range(self.num_tasks):
            # module_list.append(nn.Linear(config.hidden_size, self.num_labels))
        # self.classifier = nn.ModuleList(module_list)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels) for _ in range(self.num_tasks)])

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        """forward

        :param input_ids:
        :param labels: 给定的形式是 [batch, num_tasks]
        """

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

#         logits = []
        # for i in range(self.num_tasks):
#             logits.append(self.classifier[i](pooled_output))

        logits = [self.classifier[i](pooled_output) for i in range(self.num_tasks)]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = torch.tensor([0.])
            for i in range(self.num_tasks):
                loss += loss_fct(logits[i], labels[:,i])
            return loss 
        else:
            # 用于 验证集和测试集 标签的预测, 维度是[num_tasks, batch, num_labels]
            return torch.tensor(logits)
```
另外在 transformers 包的 \_\_init__() 中添加 MyBertForSequenceClassification :
```python
from .modeling_bert import MyBertForSequenceClassification
```
这样就可以通过下面的语句导入使用了：
```python
from transformers import MyBertForSequenceClassification
```
