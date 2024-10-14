import torch
import ssl  # 需要显式导入 ssl 模块
from record import start_audio  # 从 record.py 导入录音函数
from iat_ws_python3 import Ws_Param, websocket, on_message, on_error, on_close, finalResult,on_open  # 从 iat_ws_python3.py 导入语音识别模块
from translation import YouDaoTranslator, process_text  # 从 translation.py 导入翻译与文本处理模块
from datetime import datetime
import torch.nn as nn
import numpy as np  
from lstm import Model, vocab, pad_size, UNK, PAD, embedding_pretrained
import time

  # 录音文件名
AUDIO_FILE = "record.mp3"


# 语音识别
def recognize_audio():
    print("开始语音识别...")
    time.sleep(1)
    # 使用讯飞 API 进行语音识别
    time1 = datetime.now()
    wsParam = Ws_Param(APPID='3c84d22a', APISecret='MmIzYWMxY2ZiNjgzYzA0OTdkMzk5OGNi',
                       APIKey='ac8c29d0bb8342789b42d3b4198f7cc9',
                       AudioFile=r'record.mp3')
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = lambda ws: on_open(ws, wsParam)  # 传递 wsParam
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    time2 = datetime.now()
    print(time2 - time1)
    print("语音识别结果:", finalResult)
    return finalResult

# 翻译与清洗
def translate_and_process_text(text):
    print("开始翻译文本...")
    translator = YouDaoTranslator()
    translated_text = translator.translate(text)  
    print("翻译后的文本:", translated_text)
    # 清洗和处理翻译后的文本
    processed_tokens = [token.lower() for token in process_text(translated_text)]
    print("Processed Tokens: ", processed_tokens)
    # 将Processed Tokens连接成句子
    processed_sentence = " ".join(processed_tokens)
    # 打印连接后的句子
    print("Processed Sentence: ", processed_sentence)
    
    return processed_sentence

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

# 主流程
if __name__ == "__main__":
    # 1.录音
    start_audio(record_time=5, save_mp3=AUDIO_FILE)
    
    # 2.语音识别
    recognized_text = recognize_audio()
    
    # 3.翻译和文本处理
    processed_text = translate_and_process_text(recognized_text)
    
    # 4.文本分类
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
    input_data = process_sentence(processed_text, vocab, pad_size, PAD_id, UNK_id)

    # 进行预测
    with torch.no_grad():
        output = model(input_data)
        prediction = torch.max(output, 1)[1].cpu().numpy()

    # 输出预测结果
    predicted_label = label_map[prediction[0]]
    print(f"Predicted label for '{processed_text}' is: {predicted_label}")  