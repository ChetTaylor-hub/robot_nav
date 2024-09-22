#! /usr/bin/env python

import hashlib
import time
import uuid
from json import loads as json_loads
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# 确保已经下载了nltk的必要资源
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

# 自定义语气词列表
filler_words = {"uh", "um", "ah", "oh", "en", "ai"}

YOUDAO_URL = "https://openapi.youdao.com/api"
KEY_FILE = "./KeySecret.json"  # 存储key与secret的json文件路径
MAX_LENGTH = 1500  # 限制翻译输入的最大长度


def load_key_secret() -> tuple:
    """
    读取json文件中保存的API key
    """
    data = {
        "YouDao": {
            "APP_KEY": "6bea5586e163957c",
            "APP_SECRET": "BcCb8jIjc7gRX7JJE42Y3Q68HDsL32lz"
        }
    }
    app_key = data["YouDao"]["APP_KEY"]
    app_secret = data["YouDao"]["APP_SECRET"]
    return app_key, app_secret


class YouDaoTranslator:
    """
    调用有道翻译API实现中文翻译为英文
    """

    def __init__(self):
        self.q = ""  # 待翻译内容
        self._request_data = {}
        self._APP_KEY, self._APP_SECRET = load_key_secret()

    def _gen_sign(self, current_time: str, salt: str) -> str:
        """
        生成签名
        """
        q = self.q
        q_size = len(q)
        if q_size <= 20:
            sign_input = q
        else:
            sign_input = q[0:10] + str(q_size) + q[-10:]
        sign_str = self._APP_KEY + sign_input + salt + current_time + self._APP_SECRET
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(sign_str.encode("utf-8"))
        return hash_algorithm.hexdigest()

    def _package_data(self, current_time: str, salt: str) -> None:
        """
        设置接口调用参数
        """
        request_data = self._request_data
        request_data["q"] = self.q
        request_data["appKey"] = self._APP_KEY
        request_data["salt"] = salt
        request_data["sign"] = self._gen_sign(current_time, salt)
        request_data["signType"] = "v3"
        request_data["curtime"] = current_time
        request_data["from"] = "zh-CHS"
        request_data["to"] = "en"

    def _do_request(self) -> requests.Response:
        """
        发送请求并获取Response
        """
        current_time = str(int(time.time()))
        salt = str(uuid.uuid1())
        self._package_data(current_time, salt)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        return requests.post(YOUDAO_URL, data=self._request_data, headers=headers)

    def translate(self, q: str) -> str:
        """
        翻译中文为英文
        """
        if not q:
            return "q is empty!"
        if len(q) > MAX_LENGTH:
            return "q is too long!"

        self.q = q
        response = self._do_request()
        error_code = json_loads(response.content)["errorCode"]

        if error_code == "0":
            trans_result = json_loads(response.content)["translation"][0]  # 提取翻译结果文本
        else:
            trans_result = f"ErrorCode {error_code}, check YouDao's API doc plz."
        return trans_result


def process_text(text: str) -> list:
    """
    对翻译后的英文文本进行分词、清洗、词性标注和词形还原
    """
    # 分词，并保留标点符号
    tokens = word_tokenize(text)

    # 移除语气词，但保留标点符号
    filtered_tokens = [word for word in tokens if word.lower() not in filler_words]

    # 去除重复标点符号，保留一个
    deduplicated_tokens = []
    previous_token = ""
    for token in filtered_tokens:
        if token in {".", ",", "!", "?"} and token == previous_token:
            continue  # 跳过重复的标点符号
        deduplicated_tokens.append(token)
        previous_token = token

    # 词性标注
    pos_tags = pos_tag(deduplicated_tokens)

    # 词性还原 (Lemmatization)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]

    return lemmatized_tokens


def get_wordnet_pos(treebank_tag):
    """
    将词性标注转换为WordNet词性标注，以便词形还原
    """
    if treebank_tag.startswith('J'):
        return 'a'  # 形容词
    elif treebank_tag.startswith('V'):
        return 'v'  # 动词，动词还原为原形
    elif treebank_tag.startswith('N'):
        return 'n'  # 名词，复数还原为单数
    elif treebank_tag.startswith('R'):
        return 'r'  # 副词
    else:
        return 'n'  # 默认将其他标注当作名词处理

if __name__ == "__main__":
    translator = YouDaoTranslator()
    translated_text = translator.translate("把那根香蕉拿给我。")
    print("Translated Text: ", translated_text)

    # 处理翻译后的文本，并将所有词汇转换为小写
    processed_tokens = [token.lower() for token in process_text(translated_text)]
    print("Processed Tokens: ", processed_tokens)
    # 将Processed Tokens连接成句子
    processed_sentence = " ".join(processed_tokens)
    # 打印连接后的句子
    print("Processed Sentence: ", processed_sentence)

