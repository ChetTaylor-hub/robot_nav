import torch
import torch.nn as nn
import numpy as np
from lstm import Model, load_glove_embeddings, vocab, pad_size, UNK, PAD, embedding_pretrained

# 模型保存路径
save_path = './saved_dict/lstm.ckpt'

# 加载预训练的词向量和词表
embedding_pretrained = torch.tensor(embedding_pretrained.clone().detach())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 定义标签映射
label_map = {
    0: 'apple',
    1: 'banana',
    2: 'grape',
    3: 'orange'
}

# 确保 vocab 中有 PAD 和 UNK 的有效值
PAD_id = vocab.get(PAD, 0)  # 若 PAD 不存在，则默认使用 0
UNK_id = vocab.get(UNK, 0)  # 若 UNK 不存在，则默认使用 0

# 加载模型
model = Model().to(device)
model.load_state_dict(torch.load(save_path))
model.eval()

# 处理测试数据
def process_sentence(sentence, vocab, pad_size, PAD_id, UNK_id):
    tokenizer = lambda x: [y for y in x]  # 字符级别的tokenizer
    tokens = tokenizer(sentence)
    
    # 将单词转换为对应的词汇表 ID，若找不到则使用 UNK_id
    token_ids = [vocab.get(token, UNK_id) for token in tokens]

    # 填充或截断
    if len(token_ids) < pad_size:
        token_ids.extend([PAD_id] * (pad_size - len(token_ids)))  # 使用 PAD 进行填充
    else:
        token_ids = token_ids[:pad_size]  # 截断
    
    return torch.LongTensor([token_ids]).to(device)

# 测试数据
test_sentence = "bring me that banana ."

# 处理测试数据
input_data = process_sentence(test_sentence, vocab, pad_size, PAD_id, UNK_id)

# 进行预测
with torch.no_grad():
    output = model(input_data)
    prediction = torch.max(output, 1)[1].cpu().numpy()

# 输出预测结果
predicted_label = label_map[prediction[0]]
print(f"Predicted label for '{test_sentence}' is: {predicted_label}")
