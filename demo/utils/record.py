#! /usr/bin/env python

import pyaudio
#import keyboard
import time
from pydub import AudioSegment

def start_audio(record_time=3, save_mp3="record.mp3"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = record_time  # 需要录制的时间

    p = pyaudio.PyAudio()  # 初始化
    print(f"按下空格键开始录音...倒计时{RECORD_SECONDS}s 或按下空格提前结束。")

    # 等待空格键
    #keyboard.wait('space')
    print("开始录音...")

    # 创建录音流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    start_time = time.time()

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
  
        # 计算剩余的时间
        elapsed_time = time.time() - start_time
        remaining_time = RECORD_SECONDS - int(elapsed_time)  

        # 显示倒计时，每秒更新一次
        if i % (RATE // CHUNK) == 0:  # 每秒打印一次倒计时
            print(f"录音中，倒计时 {remaining_time} s ", end='\r')#或按下空格提前结束...

        # 检查是否按下空格键提前结束
        #if keyboard.is_pressed('space'):
        #    print("\n录音提前结束...")
        #    break

    print("\n录音结束...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将录音数据转换为MP3格式并保存
    convert_to_mp3(frames, save_mp3, RATE)

def convert_to_mp3(frames, mp3_file, rate):
    """ 将音频数据保存为 MP3 文件 """
    print(f"正在将音频数据保存为 {mp3_file}...")
    
    # 使用 pydub 将帧数据转换为音频段并导出为 MP3
    audio = AudioSegment(b''.join(frames), frame_rate=rate, sample_width=2, channels=1)
    audio.export(mp3_file, format="mp3")
    print(f"MP3 文件已保存为 {mp3_file}")

# 开始录音并保存为record.mp3
if __name__ == "__main__":
    start_audio()
  