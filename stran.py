#!/usr/bin/env python
#
# 1-D redundant forward and inverse stockwell transform code modified by GaoSong from origion st.m and ist.m code
#
# parameter:
#    data——numpy.ndarray , the origion 1-D signal
#    minfreq——int  ,  the minimum frequence in result
#    maxfreq——int  ,  the maximum frequence in result
#    samprate——int  ,  the time interval of origion signal
#    freqsamprate——int   ,  the frequence interval in result 
#    remove_edge——boolean   ,  remove a least-squares fit parabola and put a 5% hanning taper on origion signal
#    analytic_signal——boolean   ,   turn the real_value signal into analytic signal and then start s-transform
#    factor——int   ,   the width factor of localizing gaussian window: factor=1 —— original S transform
#                                                                      factor≠1 —— generalized S transform(GST)
#
# For simple use ———— if you have 500 points of 50Hz singal，you want check 10-30Hz frequency domain
#               then set st(data,minfreq=100,maxfreq=300)
#
# Date:2019/1/9
#
# Copyright (c) by GaoSong

import numpy as np
from scipy import signal 
import sys
import copy

def st(data,minfreq=0,maxfreq=None,samprate=None,freqsamprate=1,remove_edge=False,analytic_signal=False,factor=1):
    if data.shape[0] <= 1 or len(data.shape) > 1 :
        raise TypeError('input data invalid ,please check!') 
    if not maxfreq and not samprate:
        #regard signal as 1 second length
        maxfreq=len(data)//2
        samprate=len(data)
    if maxfreq and not samprate:
        samprate=len(data)
    if not maxfreq and samprate:
        maxfreq=samprate//2

    orig=copy.copy(data)
    st_res=np.zeros((int((maxfreq-minfreq)/freqsamprate)+1,len(data)),dtype='c8')	
    
    if remove_edge:
        print('remove_edge selected;  Remove trend with polynomial fit and taper!')
        try :
            from obspy.core import Trace
        except ModuleNotFoundError:
            print('Obspy not found ,please install Obspy!')
            sys.exit()
        tmp=Trace(data=orig)
        tmp.detrend('polynomial',order=2)
        tmp.taper(0.04)
        orig=tmp.data
    if analytic_signal:
        print('analytic_signal selected;  Calculating analytic signal!')
        orig=signal.hilbert(orig)

    vec=np.hstack((np.fft.fft(orig),np.fft.fft(orig)))

    if minfreq == 0:
        st_res[0]=np.mean(orig)*np.ones(len(data))
    else:
        st_res[0]=np.fft.ifft(vec[minfreq:minfreq+len(data)]*g_window(len(data),minfreq,factor))

    for i in range(freqsamprate,(maxfreq-minfreq)+1,freqsamprate):
        st_res[int(i/freqsamprate)]=np.fft.ifft(vec[minfreq+i:minfreq+i+len(data)]*g_window(len(data),minfreq+i,factor))
    return st_res


def ist(st_matrix):
    # 1-D inverse stockwell transform code modified by GaoSong from origion ist.m code
	#    the input matrix must be redundant size(N,N//2+1)
    stsp=np.sum(st_matrix,axis=1)
    if st_matrix.shape[1] % 2 != 0:
        negsp=stsp[2:].T[::-1]
    else:
        negsp=stsp[2:-1].T[::-1]

    fullstsp=np.hstack((np.conjugate(stsp.T),negsp))
    ts=np.fft.ifft(fullstsp).real
    return ts

def g_window(length,freq,factor):    
    # vector=np.zeros((2,length))
    # for i in range(length):
    #     vector[0,i]=i
    #     vector[1,i]=-length+i
    # vector=np.power(vector,2)
    # vector=vector*(-factor*2*np.pi**2/freq**2)
    # return sum(np.exp(vector))
    gauss=signal.gaussian(length,std=(freq)/(2*np.pi*factor))
    gauss=np.hstack((gauss,gauss))[length//2:length//2+length]
    return gauss

# if __name__=='__main__':

#     t=np.linspace(0,10,5001)
#     w=signal.chirp(t,f0=12.5,f1=2.5,t1=10,method='linear')
#     fmin=0
#     fmax=250
#     stres=st(w,fmin,fmax)
#     extent=(t[0],t[-1],fmin/10,fmax/10)


#     import matplotlib.pyplot as plt
    
#     fig,ax=plt.subplots(2,1)
#     ax[0].plot(t,w)
#     ax[1].imshow(np.abs(stres),origin='lower',extent=extent)
#     ax[1].axis('tight')
#     ax[1].set(xlabel='time',ylabel='frequency')
#     plt.show()\
import os
import scipy.io as scio
import matplotlib.pyplot as plt
import heartpy as hp
from PIL import Image
import scipy.signal as signal

#rootdir = r"C:\Users\86150\Desktop\xinlv\not"
#list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
"""q = 38
while q < len(list):
    path = os.path.join(rootdir, list[q])
    print(path)
    single_ecg = path
    data1 = scio.loadmat(single_ecg)
    y = data1['val'].T[1]
    y = hp.filter_signal(y,cutoff=0.1,sample_rate=1000,order=3,filtertype='notch')
    y = hp.filter_signal(y, cutoff=50, sample_rate=1000, order=3, filtertype='notch')
    y = hp.filter_signal(y, cutoff=70, sample_rate=1000, order=3, filtertype='lowpass')
    z = len(y)//2000
    #plt.plot(y)
    #plt.show()
    #wd, m = hp.process(y, 1000.0)
    #peak = wd['peaklist']
    for s in range(z):
        y2 = y[s*5000:(s+1)*5000]
        a = st(data=y2, minfreq=1, maxfreq=350, freqsamprate=1,samprate=1/1000)
        a = abs(a)
        x = np.arange(0, a.shape[1])
        c = a[:, x]
        MAS = c
        max_value = []
        min_value = []
        for iiii in range(MAS.shape[0]):
            max_value.append(max(MAS[iiii]))
            min_value.append(min(MAS[iiii]))
        max_MAS = max(max_value)
        min_MAS = min(min_value)
        k = 255 / (max_MAS - min_MAS)
        MAS = k * (MAS - min_MAS)
        #print(c.shape)
        if s < 3:
            for M_s0 in range(MAS.shape[0]):
                for M_s1 in range(MAS.shape[1]):
                    if MAS[M_s0,M_s1] <10:
                        MAS[M_s0, M_s1]=0
                    else:
                        MAS[M_s0, M_s1] = 255
            plt.imshow(MAS)
            plt.show()
            #print(MAS)
        img = Image.fromarray(MAS)
        img = img.resize([224, 224])
        image = img.convert("RGB")"""
        #image.save(r'C:\Users\86150\Desktop\py\特征提取\s变换\test\0_ecg_'+str(q+1)+'_'+str(s)+'.jpg')
        ##np.savetxt(r'C:\Users\86150\Desktop\ecg_shuju\train\0_ecg_'+str(q+1)+'_'+str(s)+'.txt',MAS)
    #q+=1

