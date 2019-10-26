## 文件说明

- pre_processing_data.py 是预处理原始文件的，将其中的不需要的特殊字符和表情和标点符号给清除，结果保存为 csv文件；

- Bert_For_MultiLabels_Classification.py 是预处理后的模型学习文件，该模块中将所有的处理和学习过程写在一起，它是 bert_model 包和 main.py 文件的总和；

- bert_model 和 main.py 是将 Bert_For_MultiLabels_Classification.py 分开写，代码更加清晰合理化；


