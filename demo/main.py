#! /usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import ssl  # 需要显式导入 ssl 模块
from utils.record import start_audio  # 从 record.py 导入录音函数
from utils.translation import YouDaoTranslator, process_text  # 从 translation.py 导入翻译与文本处理模块
import torch.nn as nn
import numpy as np  
from utils.lstm import Model, vocab, pad_size, UNK, PAD, embedding_pretrained
import time
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import warnings
warnings.filterwarnings("ignore")

# ROS相关库
import rospy  # 新增
from std_msgs.msg import String  # 新增


STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

###################################################################
# 参数设置
finalResult = ""  # 声明全局
AUDIO_FILE = "record.mp3" #录音文件名
record_time=3   #录音时长（s）
save_path = './saved_dict/lstm.ckpt'   #已训练的模型地址
###################################################################


###################################################################
# 语音识别-xunfei-API
###################################################################

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo": 1, "vad_eos": 10000}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典

        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

# 收到websocket消息的处理

def on_message(ws, message):
    global finalResult
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            # print(json.loads(message))
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
            print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
            finalResult = finalResult + result
    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        frameSize = 16000  # 每一帧的音频大小
        intervel = 0.04  # 发送音频间隔(单位:s)
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧
        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                # 文件结束
                if not buf:
                    status = STATUS_LAST_FRAME
                # 第一帧处理
                # 发送第一帧音频，带business 参数
                # appid 必须带上，只需第一帧发送
                if status == STATUS_FIRST_FRAME:
                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "lame"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # 中间帧处理
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "lame"}}
                    ws.send(json.dumps(d))
                # 最后一帧处理
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "lame"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # 模拟音频采样间隔
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())

###################################################################
# 翻译与清洗-youdao-API
###################################################################

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

###################################################################
# 使用已训练模型前的准备工作
###################################################################
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



###################################################################
# 主函数
###################################################################
if __name__ == "__main__":
    # ROS节点初始化
    rospy.init_node('speech_recognition_node', anonymous=True)  # 新增，用于初始化ROS节点
    pub = rospy.Publisher('/speech_recognition/result', String, queue_size=10)  # 新增，创建发布者，用于发布识别结果

    # 1. 录音
    start_audio(record_time, save_mp3=AUDIO_FILE)

    # 2. 语音识别
    print("开始语音识别...")
    # 使用讯飞 API 进行语音识别
    time1 = datetime.now()
    wsParam = Ws_Param(APPID='3c84d22a', APISecret='MmIzYWMxY2ZiNjgzYzA0OTdkMzk5OGNi',
                       APIKey='ac8c29d0bb8342789b42d3b4198f7cc9',
                       AudioFile=r'record.mp3')  
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    time2 = datetime.now()
    print(f"语音识别完成, 用时: {time2 - time1}")
    print("识别结果:", finalResult)  

    # 3. 翻译和文本处理
    processed_text = translate_and_process_text(finalResult)
  
    # 4. 文本分类
    embedding_pretrained = torch.tensor(embedding_pretrained.clone().detach())
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 定义标签映射
    label_map = {
        0: 'horse',
        1: 'banana',
        2: 'bus',
        3: 'bottle'
    }

    PAD_id = vocab.get(PAD, 0)
    UNK_id = vocab.get(UNK, 0)

    model = Model().to(device)
    model.load_state_dict(torch.load(save_path,map_location='cpu'))
    model.eval()

    input_data = process_sentence(processed_text, vocab, pad_size, PAD_id, UNK_id)

    # 进行预测
    with torch.no_grad():
        output = model(input_data)
        prediction = torch.max(output, 1)[1].cpu().numpy()

    predicted_label = label_map[prediction[0]]
    print(f"Predicted label for '{processed_text}' is: {predicted_label}")
    # 发布识别结果到ROS
    pub.publish(predicted_label)  # 新增，发布预测的标签
    rospy.loginfo(f"发布识别结果: {predicted_label}")  # 新增，日志记录发布的结果
