import torch
import ssl  # 需要显式导入 ssl 模块
from record import start_audio  # 从 record.py 导入录音函数
from iat_ws_python3 import Ws_Param, websocket, on_message, on_error, on_close, finalResult,on_open  # 从 iat_ws_python3.py 导入语音识别模块
from translation import YouDaoTranslator, process_text  # 从 translation.py 导入翻译与文本处理模块
from model_use import process_sentence, model, vocab, pad_size, PAD_id, UNK_id, label_map  # 从 model_use.py 导入文本分类模块
from datetime import datetime
import torch.nn as nn
import numpy as np
from lstm import Model, vocab, pad_size, UNK, PAD, embedding_pretrained

# 录音文件名
AUDIO_FILE = "record.mp3"

   
# 主流程
if __name__ == "__main__":
    # 1.录音
    start_audio(record_time=5, save_mp3=AUDIO_FILE)
    
#    