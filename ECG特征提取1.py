import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io as scio
from scipy.fftpack import fft, ifft
import math
import heartpy as hp
from PIL import Image
import stran
import time
import random



def guiyihua(data, n):
    max_value = []
    min_value = []
    for i in range(n):
        max_value.append(max(data[i]))
        min_value.append(min(data[i]))
    max_MAS = max(max_value)
    min_MAS = min(min_value)
    k = 255. / (max_MAS - min_MAS)
    data = k * (data - min_MAS)
    return data


def abcdefg(rootdir, txtpath, first, end):
    list = os.listdir(rootdir)
    # print(len(list))# 列出文件夹下所有的目录与文件
    q = first
    while q < end:
        path = os.path.join(rootdir, list[q])
        print(path)
        single_ecg = path
        data1 = scio.loadmat(single_ecg)
        single_ecg_shuju = data1['val']
        """y = data1['val'][0][0:10000]
        x = np.arange(0, len(y), 4)
        y = y[x]"""

        """ #y = hp.filter_signal(y, cutoff=1, sample_rate=200, order=3, filtertype='highpass')
        y = hp.filter_signal(y, cutoff=0.1, sample_rate=500, order=3, filtertype='notch')
        y = hp.filter_signal(y, cutoff=50, sample_rate=500, order=3, filtertype='notch')
        y = hp.filter_signal(y, cutoff=70, sample_rate=500, order=3, filtertype='lowpass')
        #plt.plot(y)
        #plt.show()
        wd, m = hp.process(y, sample_rate=500)
        peak = wd['peaklist']
        peak = np.array(peak[1:len(peak) - 2]).tolist()"""
        peak_random = 1 #random.sample(peak, 1)
        for peaki in range(peak_random):
            # b = y[peak_random[peaki] - 100:peak_random[peaki] + 150]
            # print(peak_random[peaki])
            # plt.plot(b)
            # plt.show()
            # for ii in range(1):
            i = 0
            MAS = []
            T_BD = []
            MEAN = []
            #FEATURE_ECG = []
            S_FEATURE = []
            while i < 12:
                b = single_ecg_shuju[i][0:10000]

                low = np.arange(0, len(b), 4)
                b = b[low]
                # print(len(b))
                rate = 125
                # b111, a111 = signal.butter(6, 0.4/rate, 'highpass')
                # b = signal.filtfilt(b111, a111, b)
                y2 = signal.medfilt(b, 125)
                b = b - y2
                b1, a1 = signal.butter(6, 70 / rate, 'lowpass')
                b = signal.filtfilt(b1, a1, b)
                b2, a2 = signal.butter(6, [48 / rate, 52 / rate], 'bandstop')
                b = signal.filtfilt(b2, a2, b)
                #plt.plot(b)
                #plt.show()
                #s_b = stran.st(b, minfreq=0, maxfreq=500, freqsamprate=1, samprate=1)
                #s_b = abs(s_b)

                #s_b_b = np.sum(s_b, axis=0) / s_b.shape[0]

                #b = b[peak_random[peaki] - 200:peak_random[peaki] + 300]
                #feature_ecg = b
                #FEATURE_ECG.append(feature_ecg)

                for t in range(1, 2):
                    y = b
                    """s_b = stran.st(y, minfreq=0, maxfreq=1000, freqsamprate=1, samprate=1)
                    s_b = abs(s_b)
                    #plt.imshow(s_b)
                    #plt.show()
                    s_b_b = np.sum(s_b, axis=0) / s_b.shape[0]
                    # plt.plot(s_b_b)
                    # plt.show()
                    S_FEATURE.append(s_b_b)
                    s_b_mean = sum(s_b_b) / len(y)
                    s_sig = 0
                    s_rms = 0
                    s_b_bd = 0
                    for bodong1 in range(1, len(y)):
                        s_b_bd = s_b_bd + abs(s_b_b[bodong1] - s_b_b[bodong1 - 1])
                        s_sig = s_sig + (s_b_b[bodong1] - s_b_mean) ** 2
                        s_rms = s_rms + (s_b_b[bodong1]) ** 2
                    s_b_bd = s_b_bd
                    s_sigma = s_sig / len(y)
                    s_X_rms = (s_rms / len(y)) ** 0.5"""
                    # print(s_b_bd,s_sigma,s_X_rms)
                    # print(s_b_bd)
                    # plt.imshow(abs(s_b))

                    # img = Image.fromarray(abs(s_b))
                    # image = img.convert("RGB")
                    # image.save(txtpath + str(q+1)+'_'+str(i+1) +'导联_'+str(ii)+ '.jpg')
                    # plt.title(str(i+1))
                    # plt.show()
                    # MEAN.append(sum(y)/len(y))
                    y3_ = y
                    """bd = 0
                    sig = 0
                    rms = 0
                    mean = sum(y) / len(y)
                    for bodong in range(1, len(y)):
                        bd = bd + abs(y[bodong] - y[bodong - 1])
                        sig = sig + (y[bodong] - mean) ** 2
                        rms = rms + (y[bodong]) ** 2
                    bd = bd / len(y)
                    sigma = sig / len(y) / 500
                    X_rms = (rms / len(y)) ** 0.5"""

                    js = 3
                    bb1, aa1 = signal.butter(js, [0.5 / rate, 3 / rate], 'bandpass')
                    ztqq1 = signal.filtfilt(bb1, aa1, y3_)
                    bb2, aa2 = signal.butter(js, [2 / rate, 5 / rate], 'bandpass')
                    ztqq2 = signal.filtfilt(bb2, aa2, y3_)
                    bb3, aa3 = signal.butter(js, [4 / rate, 7 / rate], 'bandpass')
                    ztqq3 = signal.filtfilt(bb3, aa3, y3_)
                    fft_ztqq1 = fft(ztqq1)
                    ztqq1_f = abs(fft_ztqq1)
                    fft_ztqq2 = fft(ztqq2)
                    ztqq2_f = abs(fft_ztqq2)
                    fft_ztqq3 = fft(ztqq3)
                    ztqq3_f = abs(fft_ztqq3)
                    mas01 = sum(ztqq1_f) / len(ztqq1_f)
                    mas02 = sum(ztqq2_f) / len(ztqq2_f)
                    mas03 = sum(ztqq3_f) / len(ztqq3_f)
                    b3, a3 = signal.butter(js, [6 / rate, 9 / rate], 'bandpass')
                    y4_3_15 = signal.filtfilt(b3, a3, y3_)
                    b4, a4 = signal.butter(js, [8 / rate, 11 / rate], 'bandpass')
                    y4_15_35 = signal.filtfilt(b4, a4, y3_)
                    b5, a5 = signal.butter(js, [10 / rate, 13 / rate], 'bandpass')
                    y4_35_49 = signal.filtfilt(b5, a5, y3_)
                    b6, a6 = signal.butter(js, [12 / rate, 15 / rate], 'bandpass')
                    y4_51_70 = signal.filtfilt(b6, a6, y3_)
                    b7, a7 = signal.butter(js, [14 / rate, 17 / rate], 'bandpass')
                    y4_70_90 = signal.filtfilt(b7, a7, y3_)
                    b8, a8 = signal.butter(js, [16 / rate, 19 / rate], 'bandpass')
                    y4_90_120 = signal.filtfilt(b8, a8, y3_)
                    b9, a9 = signal.butter(js, [18 / rate, 21 / rate], 'bandpass')
                    y4_120_150 = signal.filtfilt(b9, a9, y3_)
                    b10, a10 = signal.butter(js, [20 / rate, 23 / rate], 'bandpass')
                    y4_150_200 = signal.filtfilt(b10, a10, y3_)
                    b11, a11 = signal.butter(js, [22 / rate, 25 / rate], 'bandpass')
                    ztq1 = signal.filtfilt(b11, a11, y3_)
                    b12, a12 = signal.butter(js, [24 / rate, 27 / rate], 'bandpass')
                    ztq2 = signal.filtfilt(b12, a12, y3_)
                    b13, a13 = signal.butter(js, [26 / rate, 29 / rate], 'bandpass')
                    ztq3 = signal.filtfilt(b13, a13, y3_)
                    b14, a14 = signal.butter(js, [28 / rate, 31 / rate], 'bandpass')
                    ztq4 = signal.filtfilt(b14, a14, y3_)
                    b15, a15 = signal.butter(js, [30 / rate, 33 / rate], 'bandpass')
                    ztq5 = signal.filtfilt(b15, a15, y3_)
                    b16, a16 = signal.butter(js, [32 / rate, 35 / rate], 'bandpass')
                    ztq6 = signal.filtfilt(b16, a16, y3_)
                    b17, a17 = signal.butter(js, [34 / rate, 37 / rate], 'bandpass')
                    ztq7 = signal.filtfilt(b17, a17, y3_)
                    b18, a18 = signal.butter(js, [36 / rate, 39 / rate], 'bandpass')
                    ztq8 = signal.filtfilt(b18, a18, y3_)
                    b19, a19 = signal.butter(js, [38 / rate, 41 / rate], 'bandpass')
                    ztq9 = signal.filtfilt(b19, a19, y3_)
                    b20, a20 = signal.butter(js, [40 / rate, 48 / rate], 'bandpass')
                    ztq10 = signal.filtfilt(b20, a20, y3_)
                    b21, a21 = signal.butter(js, [52 / rate, 70 / rate], 'bandpass')
                    ztq11 = signal.filtfilt(b21, a21, y3_)

                    fft_ztq1 = fft(ztq1)
                    ztq1_f = abs(fft_ztq1)

                    fft_ztq2 = fft(ztq2)
                    ztq2_f = abs(fft_ztq2)

                    fft_ztq3 = fft(ztq3)
                    ztq3_f = abs(fft_ztq3)

                    fft_ztq4 = fft(ztq4)
                    ztq4_f = abs(fft_ztq4)

                    fft_ztq5 = fft(ztq5)
                    ztq5_f = abs(fft_ztq5)

                    fft_ztq6 = fft(ztq6)
                    ztq6_f = abs(fft_ztq6)

                    fft_ztq7 = fft(ztq7)
                    ztq7_f = abs(fft_ztq7)

                    fft_ztq8 = fft(ztq8)
                    ztq8_f = abs(fft_ztq8)

                    fft_ztq9 = fft(ztq9)
                    ztq9_f = abs(fft_ztq9)

                    fft_ztq10 = fft(ztq10)
                    ztq10_f = abs(fft_ztq10)

                    fft_ztq11 = fft(ztq11)
                    ztq11_f = abs(fft_ztq11)

                    fft_y4_3_15 = fft(y4_3_15)
                    y4_3_15f = abs(fft_y4_3_15)

                    fft_y4_15_35 = fft(y4_15_35)
                    y4_15_35f = abs(fft_y4_15_35)

                    fft_y4_35_49 = fft(y4_35_49)
                    y4_35_49f = abs(fft_y4_35_49)

                    fft_y4_51_70 = fft(y4_51_70)
                    y4_51_70f = abs(fft_y4_51_70)

                    fft_y4_70_90 = fft(y4_70_90)
                    y4_70_90f = abs(fft_y4_70_90)

                    fft_y4_90_120 = fft(y4_90_120)
                    y4_90_120f = abs(fft_y4_90_120)

                    fft_y4_120_150 = fft(y4_120_150)
                    y4_120_150f = abs(fft_y4_120_150)

                    fft_y4_150_200 = fft(y4_150_200)
                    y4_150_200f = abs(fft_y4_150_200)

                    mas11 = sum(y4_3_15f) / len(y4_3_15f)
                    mas12 = sum(y4_15_35f) / len(y4_15_35f)
                    mas13 = sum(y4_35_49f) / len(y4_15_35f)
                    mas14 = sum(y4_51_70f) / len(y4_51_70f)
                    mas15 = sum(y4_70_90f) / len(y4_70_90f)
                    mas16 = sum(y4_90_120f) / len(y4_90_120f)
                    mas17 = sum(y4_120_150f) / len(y4_120_150f)
                    mas18 = sum(y4_150_200f) / len(y4_150_200f)
                    mas19 = sum(ztq1_f) / len(ztq1_f)
                    mas20 = sum(ztq2_f) / len(ztq2_f)
                    mas21 = sum(ztq3_f) / len(ztq3_f)
                    mas22 = sum(ztq4_f) / len(ztq4_f)
                    mas23 = sum(ztq5_f) / len(ztq5_f)
                    mas24 = sum(ztq6_f) / len(ztq6_f)
                    mas25 = sum(ztq7_f) / len(ztq7_f)
                    mas26 = sum(ztq8_f) / len(ztq8_f)
                    mas27 = sum(ztq9_f) / len(ztq9_f)
                    mas28 = sum(ztq10_f) / len(ztq10_f)
                    mas29 = sum(ztq11_f) / len(ztq11_f)

                    """featur = np.array([mean, sigma, X_rms, bd, s_b_mean, s_sigma, s_X_rms, s_b_bd])
                    fea1 = max(featur)
                    fea2 = min(featur)
                    kk = 255 / (fea1 - fea2)
                    featur = kk * (featur - fea2)*3"""

                    """mas1 = [mas01, mas02, mas03, mas11, mas12, mas13, mas14, mas15, mas16, mas17, mas18, mas19, mas20,
                            mas21, mas22, mas23, mas24, mas25, mas26, mas27, mas28, mas29,
                            featur[0], featur[1], featur[2], featur[3],
                            featur[4], featur[5], featur[6], featur[7]]"""
                    mas1 = [mas01, mas02, mas03, mas11, mas12, mas13, mas14, mas15, mas16, mas17, mas18, mas19, mas20,
                            mas21, mas22, mas23, mas24, mas25, mas26, mas27, mas28, mas29]
                    # MAS.append(mas1)
                    # MAS_Z.append(MAS)
                    MAS.append(mas1)


                i += 1
            # print(MEAN)

            """波动指数打印"""
            """print()
            print(sum(T_BD))
            print(sum(T_BD)/len(T_BD))
            print()"""
            """0-255归一化"""
            """FEATURE_ECG = np.array(FEATURE_ECG)
            for g in range(FEATURE_ECG.shape[0]):
                k1 = 255 / (max(FEATURE_ECG[g]) - min(FEATURE_ECG[g]))
                FEATURE_ECG[g] = k1 * (FEATURE_ECG[g] - min(FEATURE_ECG[g]))


            #FEATURE_ECG = guiyihua(FEATURE_ECG,12)
            print(FEATURE_ECG)
            S_FEATURE = np.array(S_FEATURE)
            S_FEATURE = guiyihua(S_FEATURE,12)

            for f in range(FEATURE_ECG.shape[0]):
                plt.plot(FEATURE_ECG[f])
                plt.show()"""
            MAS = np.array(MAS)
            MAS = guiyihua(MAS, 12)
            #print(MAS.shape)
            image = Image.fromarray(MAS)
            image = image.resize([22, 22])
            image = image.convert("L")
            #plt.imshow(image)
            #plt.show()
            #print(MAS)

            """#print(MAS.shape)
            img = Image.fromarray(MAS)
            image = img.resize([224,12])
            MAS = np.asarray(image)
            feature = np.row_stack((np.row_stack((FEATURE_ECG,S_FEATURE)), MAS))
            #feature = guiyihua(feature,36)
            img = Image.fromarray(feature)
            image = img.resize([224, 224])
            image = image.convert("RGB")


            #print(image)
            plt.imshow(image)
            plt.show()"""
            image.save(txtpath + str(q + 1) + '_' + str(peaki + 1) + '.jpg')
            # np.savetxt(txtpath+str(q+1)+'_'+str(peaki+1)+'.txt', MAS)
        q += 1


start = time.time()
rootdir = r"C:\Users\1\Desktop\PTBdatabase\PTBmat8\0"
abcdefg(rootdir, r'C:\Users\1\Desktop\PTBdatabase\PTBmat8\MAS0\0_ecg_', 0, 80)
end1 = time.time()
print(f'历时{(end1 - start) / 60}分钟')
print(f'历时{(end1 - start) / 60 / 60}小时')
rootdir = r"C:\Users\1\Desktop\PTBdatabase\PTBmat8\1"
abcdefg(rootdir, r'C:\Users\1\Desktop\PTBdatabase\PTBmat8\MAS1\1_ecg_', 0, 366)
end2 = time.time()
print(f'历时{(end2 - start) / 60}分钟')
print(f'历时{(end2 - start) / 60 / 60}小时')
rootdir = r"C:\Users\1\Desktop\PTBdatabase\PTBmat8\2"
abcdefg(rootdir, r'C:\Users\1\Desktop\PTBdatabase\PTBmat8\MAS2\2_ecg_', 0, 20)
end3 = time.time()
print(f'历时{(end3 - start) / 60}分钟')
print(f'历时{(end3 - start) / 60 / 60}小时')
rootdir = r"C:\Users\1\Desktop\PTBdatabase\PTBmat8\3"
abcdefg(rootdir, r'C:\Users\1\Desktop\PTBdatabase\PTBmat8\MAS3\3_ecg_', 0, 16)
end = time.time()
print(f'历时{(end - start) / 60}分钟')
print(f'历时{(end - start) / 60 / 60}小时')
rootdir = r"C:\Users\1\Desktop\PTBdatabase\PTBmat8\4"
abcdefg(rootdir, r'C:\Users\1\Desktop\PTBdatabase\PTBmat8\MAS4\4_ecg_', 0, 17)