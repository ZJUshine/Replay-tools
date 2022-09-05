import librosa
import numpy as np
import soundfile as sf
import librosa.displayc
import matplotlib as plt

##########  拼接音频  ##########
origin_tts_path = './origin_tts.wav'
audio_1k_path = './1k.wav'

y_origin_tts, sr_origin_tts = librosa.load(origin_tts_path,sr = None)
y_1k, sr_1k = librosa.load(audio_1k_path,sr = None)
if (sr_origin_tts != sr_1k):
    print("Warning:sr is different")

origin_1k_tts = np.hstack((y_1k, y_origin_tts))
sf.write('./audio_1k_tts.wav', origin_1k_tts,sr_origin_tts)


###### 音频切割 #######
RANGES = 91
def cut_pYin(file_path,number=1,dur=2):
    y, sr = librosa.load(file_path)
    #最大频率和最小频率至少相差800Hz
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=990, fmax=1790)
    times = librosa.times_like(f0)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()
    cut_time = []

    for index in range(len(f0)-1):
        if (np.isnan(f0[index]) and not(np.isnan(f0[index+1]))):
            result = True
            for test_index in range (index+1,index+RANGES):
                result = result and not(np.isnan(f0[test_index]))
            if (result == True):
                # 1k 音频长为5.5s
                cut_time.append(times[index + 1] + 5.5)
    if(len(cut_time) >= number):
        print("success")
        for wav_index in range (number):
            start_time = cut_time[wav_index]
            stop_time = start_time + dur
            audio_dst = y[int(start_time * sr):int(stop_time * sr)]
            sf.write('./replay'+f'_{wav_index}.wav', audio_dst, sr)
        print(cut_time)
    else:
        print ("error")
# number 为重放次数 dur 为原音频不加1k信号长度
cut_pYin("./replay_1k_tts.wav", number=1, dur=1.82)