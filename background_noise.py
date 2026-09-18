# -*- coding: utf-8 -*-
"""
Created on Tues Aug 18 13:52:30 2026

@author: pelet
"""

import pandas as pd
from datetime import datetime, timedelta
from haversine import haversine,Unit
import math
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import obspy
import os
from numpy import trapezoid

#constant

shotpath = r"D:\Survey_reports\Shotlog.xlsx"
sac_file_folder_path = r"D:\OBS Raw Data"
output_folder = r"D:\Spatial Analysis\Extra_sound_metrics"
predictions_folder = r"D:\Spatial Analysis\Extra_sound_metrics"

upa_per_count =48.25

#########
####Martin Code
import numpy as np
from scipy.special import jv

def bessel(v, X):
    return ((1j**(-v))*jv(v,1j*X)).real

def stft(x, n_fft=512, win_length=400, hop_length=160, window='hamming'):   
    if window == 'hanning':
        window = np.hanning(win_length)
    elif window == 'hamming':
        window = np.hamming(win_length)
    elif window == 'rectangle':
        window = np.ones(win_length)
    return np.array([np.fft.rfft(window*x[i:i+win_length],n_fft,axis=0) for i in range(0, len(x)-win_length, hop_length)])

def estnoisem(pSpectrum,hop_length):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
    	 Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    
    (nFrames,nFFT2)=np.shape(pSpectrum)          # number of frames and freq bins
    x=np.array(np.zeros((nFrames,nFFT2)) )           # initialize output arrays
    xs=np.array(np.zeros((nFrames,nFFT2)) )           # will hold std error in the future

    # default algorithm constants
    taca= 0.0449    # smoothing time constant for alpha_c = -hop_length/log(0.7) in equ (11)
    tamax= 0.392    # max smoothing time constant in (3) = -hop_length/log(0.96)
    taminh= 0.0133    # min smoothing time constant (upper limit) in (3) = -hop_length/log(0.3)
    tpfall= 0.064   # time constant for P to fall (12)
    tbmax= 0.0717   # max smoothing time constant in (20) = -hop_length/log(0.8)
    qeqmin= 2.0       # minimum value of Qeq (23)
    qeqmax= 14.0      # max value of Qeq per frame
    av= 2.12             # fudge factor for bc calculation (23 + 13 lines)
    td= 1.536       # time to take minimum over
    nu= 8          # number of subwindows
    qith= np.array([0.03, 0.05, 0.06, np.inf],dtype=float) # noise slope thresholds in dB/s
    nsmdb= np.array([47, 31.4, 15.7, 4.1],dtype=float) # maximum permitted +ve noise slope in dB/s


    # derived algorithm constants
    aca=np.exp(-hop_length/taca) # smoothing constant for alpha_c in equ (11) = 0.7
    acmax=aca          # min value of alpha_c = 0.7 in equ (11) also = 0.7
    amax=np.exp(-hop_length/tamax) # max smoothing constant in (3) = 0.96
    aminh=np.exp(-hop_length/taminh) # min smoothing constant (upper limit) in (3) = 0.3
    bmax=np.exp(-hop_length/tbmax) # max smoothing constant in (20) = 0.8
    SNRexp = -hop_length/tpfall
    nv=round(td/(hop_length*nu))    # length of each subwindow in frames


    if nv<4:        # algorithm doesn't work for miniscule frames
        nv=4
        nu=round(td/(hop_length*nv))
    nd=nu*nv           # length of total window in frames
    (md,hd,dd) = mhvals(nd) # calculate the constants M(D) and H(D) from Table III
    (mv,hv,dv) = mhvals(nv) # calculate the constants M(D) and H(D) from Table III
    nsms=np.array([10])**(nsmdb*nv*hop_length/10)  # [8 4 2 1.2] in paper
    qeqimax=1/qeqmin  # maximum value of Qeq inverse (23)
    qeqimin=1/qeqmax # minumum value of Qeq per frame inverse
    

    p=pSpectrum[0,:]         # smoothed power spectrum
    ac=1               # correction factor (9)
    sn2=p              # estimated noise power
    pb=p               # smoothed noisy speech power (20)
    pb2=pb**2
    pminu=p
    actmin=np.array(np.ones(nFFT2) * np.inf)   # Running minimum estimate
    actminsub=np.array(np.ones(nFFT2) * np.inf)          # sub-window minimum estimate
    subwc=nv                   # force a buffer switch on first loop
    actbuf=np.array(np.ones((nu,nFFT2)) * np.inf)  # buffer to store subwindow minima
    ibuf=0
    lminflag=np.zeros(nFFT2)      # flag to remember local minimum

    # loop for each frame
    for t in range(0,nFrames):                       # we use t instead of lambda in the paper
        pSpectrum_t=pSpectrum[t,:]                      # noise speech power spectrum
        acb=(1+(sum(p) / sum(pSpectrum_t)-1)**2)**(-1)    # alpha_c-bar(t)  (9)
        
        tmp=np.array([acb] )
        tmp[tmp < acmax] = acmax
        #max_complex(np.array([acb] ),np.array([acmax] ))
        
        ac=aca*ac+(1-aca)*tmp    # alpha_c(t)  (10)
        
        ah=amax*ac*(1+(p/sn2-1)**2)**(-1)       # alpha_hat: smoothing factor per frequency (11)
        SNR=sum(p)/sum(sn2)
        
               
        ah=max_complex(ah,min_complex(np.array([aminh] ),np.array([SNR**SNRexp] )))                            # lower limit for alpha_hat (12)

        p=ah*p+(1-ah)*pSpectrum_t            # smoothed noisy speech power (3)

        b=min_complex(ah**2,np.array([bmax] )) # smoothing constant for estimating periodogram variance (22 + 2 lines)
        pb=b*pb + (1-b)*p            # smoothed periodogram (20)
        pb2=b*pb2 + (1-b)*p**2     	 # smoothed periodogram squared (21)

        qeqi=max_complex(min_complex((pb2-pb**2)/(2*sn2**2),np.array([qeqimax] )),np.array([qeqimin/(t+1)] ))   # Qeq inverse (23)
        qiav=sum(qeqi)/nFFT2              # Average over all frequencies (23+12 lines) (ignore non-duplication of DC and nyquist terms)
        bc=1+av*np.sqrt(qiav)              # bias correction factor (23+11 lines)
        bmind=1+2*(nd-1)*(1-md)/(qeqi**(-1)-2*md)      # we use the signalmplified form (17) instead of (15)
        bminv=1+2*(nv-1)*(1-mv)/(qeqi**(-1)-2*mv)    # same expressignalon but for sub windows
        kmod=(bc*p*bmind) < actmin      # Frequency mask for new minimum

        if any(kmod):
            actmin[kmod]=bc*p[kmod]*bmind[kmod]
            actminsub[kmod]=bc*p[kmod]*bminv[kmod]

        if subwc>1 and subwc<nv:              # middle of buffer - allow a local minimum
            lminflag= np.logical_or(lminflag,kmod)    	# potential local minimum frequency bins
            pminu=min_complex(actminsub,pminu)
            sn2=pminu.copy()
        else:
            if subwc>=nv:                   # end of buffer - do a buffer switch
                ibuf=1+(ibuf%nu)     	# increment actbuf storage pointer
                actbuf[ibuf-1,:]=actmin.copy()   	# save sub-window minimum
                pminu=min_complex_mat(actbuf)
                i=np.nonzero(np.array(qiav )<qith)
                nsm=nsms[i[0][0]]     	# noise slope max
                lmin = np.logical_and(np.logical_and(np.logical_and(lminflag,np.logical_not(kmod)),actminsub<(nsm*pminu)),actminsub>pminu)
                if any(lmin):
                    pminu[lmin]=actminsub[lmin]
                    actbuf[:,lmin]= np.ones((nu,1)) * pminu[lmin]
                lminflag[:]=0
                actmin[:]=np.inf
                subwc=0

        subwc=subwc+1
        x[t,:]=sn2.copy()
        qisq=np.sqrt(qeqi)
        # empirical formula for standard error based on Fig 15 of [2]
        xs[t,:]=sn2*np.sqrt(0.266*(nd+100*qisq)*qisq/(1+0.005*nd+6/nd)/(0.5*qeqi**(-1)+nd-1))


    return x

def mhvals(*args):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    nargin = len(args)

    dmh=np.array([
        [1,   0,       0],
        [2,   0.26,    0.15],
        [5,   0.48,    0.48],
        [8,   0.58,    0.78],
        [10,  0.61,    0.98],
        [15,  0.668,   1.55],
        [20,  0.705,   2],
        [30,  0.762,   2.3],
        [40,  0.8,     2.52],
        [60,  0.841,   3.1],
        [80,  0.865,   3.38],
        [120, 0.89,    4.15],
        [140, 0.9,     4.35],
        [160, 0.91,    4.25],
        [180, 0.92,    3.9],
        [220, 0.93,    4.1],
        [260, 0.935,   4.7],
        [300, 0.94,    5]
        ],dtype=float)

    if nargin>=1:
        d=args[0]
        i=np.nonzero(d<=dmh[:,0])
        if len(i)==0:
            i=np.shape(dmh)[0]-1
            j=i
        else:
            i=i[0][0]
            j=i-1
        if d==dmh[i,0]:
            m=dmh[i,1]
            h=dmh[i,2]
        else:
            qj=np.sqrt(dmh[i-1,0])    # interpolate usignalng sqrt(d)
            qi=np.sqrt(dmh[i,0])
            q=np.sqrt(d)
            h=dmh[i,2]+(q-qi)*(dmh[j,2]-dmh[i,2])/(qj-qi)
            m=dmh[i,1]+(qi*qj/q-qj)*(dmh[j,1]-dmh[i,1])/(qi-qj)
    else:
        d=dmh[:,0].copy()
        m=dmh[:,1].copy()
        h=dmh[:,2].copy()

    return m,h,d


def max_complex(a,b):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    if len(a)==1 and len(b)>1:
        a=np.tile(a,np.shape(b))
    if len(b)==1 and len(a)>1:
        b=np.tile(b,np.shape(a))

    i=np.logical_or(np.iscomplex(a),np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i]=np.absolute(aa[i])
        bb[i]=np.absolute(bb[i])
    if a.dtype == 'complex' or b.dtype== 'complex':
        cc = np.array(np.zeros(np.shape(a)) )
    else:
        cc = np.array(np.zeros(np.shape(a)),dtype=float)

    i=aa>bb
    cc[i]=a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc

def min_complex(a,b):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    if len(a)==1 and len(b)>1:
        a=np.tile(a,np.shape(b))
    if len(b)==1 and len(a)>1:
        b=np.tile(b,np.shape(a))

    i=np.logical_or(np.iscomplex(a),np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i]=np.absolute(aa[i])
        bb[i]=np.absolute(bb[i])

    if a.dtype == 'complex' or b.dtype== 'complex':
        cc = np.array(np.zeros(np.shape(a)) )
    else:
        cc = np.array(np.zeros(np.shape(a)),dtype=float)

    i=aa<bb
    cc[i]=a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc

def min_complex_mat(a):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    s=np.shape(a)
    m = np.array(np.zeros(s[1]) )
    for i in range(0,s[1]):
        j = np.argmin(np.absolute(a[:,i]))
        m[i] = a[j,i]
    return m



#################################
####### My Functions from spatialpattern.py

def read_predictions_file(predictions_path):
    predictions = pd.read_csv(predictions_path)

    #correct datetimes (-30 mins from each)
    predictions["corrected_datetime"] = pd.to_datetime(predictions["file_datetime"],format="%Y-%m-%dT%H:%M:%S") - pd.Timedelta(minutes=30)
    predictions["corrected_datetime"] = predictions["corrected_datetime"] + pd.to_timedelta(predictions["clip_start_sec"],unit="s")


    return predictions

def get_station_from_filename(filename):
    """
    Extract station name from filenames like:
    OBH04_R160.01.sac
    OBS11_R189.01.SAC
    OBS9_R200.01.sac
    OBS08_R200.01.sac
    """
    match = re.search(r"(OBH|OBS)\d+", filename.upper())
    return match.group(0) if match else None


def get_shot_datetime(year, julian_day, time_value):
   #get a datetime from the shot log columns

    year = int(year)
    julian_day = int(julian_day)

    date_at_start_of_year = datetime(year, 1, 1)
    shot_date = date_at_start_of_year + timedelta(days=julian_day - 1)

    # get time from excel
    if hasattr(time_value, "hour"):
        shot_datetime = shot_date + timedelta(
            hours=time_value.hour,
            minutes=time_value.minute,
            seconds=time_value.second,
            microseconds=time_value.microsecond
        )

    return shot_datetime

def read_shot_file(shot_path):

    shots = pd.read_excel(shot_path, header=None)

    shots = shots.iloc[:, [0,1,2,3,4,5]].copy()

    shots.columns = [
        "shot_number",
        "year",
        "julian_day",
        "shot_times",
        "latitude",
        "longitude",
    ]

    shot_times = []

    for i in range(len(shots)):
        shot_time = get_shot_datetime(shots.loc[i, "year"],shots.loc[i, "julian_day"],shots.loc[i, "shot_times"])

        shot_times.append(shot_time)

    shots["shot_time"] = shot_times

    #sorts shot times in chronological order (if they arent already)
    shots = shots.sort_values("shot_time").reset_index(drop=True)

    print("Shot file:", shot_path,"Number of shots:", len(shots),"First shot:", shots["shot_time"].min(),"Last shot:", shots["shot_time"].max())

    return shots


def get_sac_from_wav_name(filename,segment_start):
    
    match = re.search(r"(OBH|OBS)\d+", filename.upper())
    instrument = match.group(0) if match else None
    
    segment_start = pd.Timestamp(segment_start)
    julian_day = segment_start.dayofyear
    
    sac_path_end = f"{instrument}.JD{julian_day:03d}.CH3.SAC"
    
    sac_path = sac_file_folder_path + '\\' + sac_path_end
    
    print(sac_path)
    return sac_path



def get_segment_pressures(current_segment_start,current_segment_end, sac_path, upa_per_count):
    #function to find pressures for the given segment using calibration data

    st = obspy.read(str(sac_path))
    tr = st[0]

    sac_start_time = tr.stats.starttime.datetime
    sampling_rate = tr.stats.sampling_rate
    dt = 1.0 / sampling_rate
    
    #find the index of the start and end of the segment in the sac file
    delta_t = timedelta(seconds=dt)
    timeshift = current_segment_start - sac_start_time
    end_timeshift = current_segment_end - sac_start_time
    
    start_sample = int(timeshift / delta_t)
    end_sample = int(end_timeshift / delta_t)
    
    segment_counts = tr.data[start_sample:end_sample].astype(float)

    pressure_uPa = segment_counts * upa_per_count
    pressure_Pa = pressure_uPa / 1e6

    print("SAC file:", sac_path,"Start time:", sac_start_time,"Segment start", current_segment_start, "Number of samples:", len(pressure_Pa))
    
    return pressure_Pa, sac_start_time, sampling_rate, dt




##############
##### New function to use martin code

def get_noise_floor(pressure_frame,sampling_rate,n_fft=512,win_length=2,hop_length=0.1):
    
    #check the size of the pressure frame
    if len(pressure_frame) < win_length:
        return np.nan
    
### Remove the mean before fourier transform
    pressure_frame = pressure_frame - np.mean(pressure_frame)
    
### convert window and hop lengths to number of samples for fft
    win_length_sample = round(win_length * sampling_rate)
    hop_length_sample = round(hop_length * sampling_rate)
    
    ### use martins stft function
    f_coeffs = stft(pressure_frame,n_fft=n_fft, win_length=win_length_sample, hop_length=hop_length_sample, window='hamming')
    
    pSpectrum = np.abs(f_coeffs) ** 2
    
    window = np.hamming(win_length_sample)

    pSpectrum = pSpectrum/(sampling_rate * np.sum(window ** 2))
    
    estNoise = estnoisem(pSpectrum, hop_length)
    
    return estNoise
    
    
## Test function on a wav file

pressure_Pa, sac_start_time, sampling_rate, dt = get_segment_pressures(current_segment_start=pd.Timestamp('06/06/2013 13:20'),current_segment_end=pd.Timestamp('06/06/2013 13:25'), sac_path=r"D:\OBS Raw Data\OBS21.JD157.CH3.SAC", upa_per_count=48.25)

noise_est = get_noise_floor(pressure_Pa,sampling_rate,n_fft=512,win_length=2,hop_length=0.1)

print(noise_est.shape)

# find the mean of all the time windows
mean_PSD = np.mean(noise_est,axis=0)
print(mean_PSD.shape)
mean_PSD_dB = 10 * np.log10(mean_PSD / (1e-6 ** 2))

#frequencies in the PSD axis
frequencies = np.linspace(0,sampling_rate / 2,noise_est.shape[1])
print(frequencies.shape)

#filter to 18-34Hz
low_freq= 18
high_freq = 34
frequencies_filter = ((frequencies >= low_freq)& (frequencies <= high_freq))

bg= "#F1F0EB"
plt.figure(figsize=(10, 4), facecolor=bg)
plt.gca().set_facecolor(bg)
plt.plot(frequencies,mean_PSD_dB)
plt.xlabel("Frequency (Hz)",fontsize=14)
plt.ylabel("Mean power spectral density\n (dB re μPa$^2$ /Hz)",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()#plt.title("Power Spectral Density")


#integrate area under curve
area = trapezoid(mean_PSD[frequencies_filter]*2, frequencies[frequencies_filter])
area = 10 * np.log10(area / (1e-6 ** 2))
print("area =", area) 



def process_5_min_segments(shots,predictions):
    results = []
    first_shot_time = pd.Timestamp(shots["shot_time"].min())
    first_segment_start = first_shot_time.round(freq='5min')
    last_segment_end = pd.Timestamp(shots["shot_time"].max()).ceil('5min')
    #first_segment_end = first_segment_start + pd.Timedelta(minutes=4,seconds=59)
    
    current_segment_start = first_segment_start
    current_segment_end = current_segment_start + pd.Timedelta(minutes =5)
    
    while current_segment_end <= last_segment_end:
        
        #get only predictions in this timeframe
        segment_predictions = []
        segment_predictions = (predictions["corrected_datetime"] >= current_segment_start) & (predictions["corrected_datetime"] < current_segment_end)
        segment_predictions = predictions[segment_predictions].copy()
        
        #check if the predictions file ends before the shooting file (if OBS stopped recording) and skip segment if it does
        if len(segment_predictions) == 0:
            print("No predictions for:", current_segment_start, "to", current_segment_end)
        
            current_segment_start += pd.Timedelta(minutes=5)
            current_segment_end += pd.Timedelta(minutes=5)
            continue

        #get number of shots
        segment_shots = []
        segment_shots = (shots["shot_time"] >= current_segment_start) & (shots["shot_time"] < current_segment_end)
        segment_shots = shots[segment_shots].copy()
        number_of_shots = len(segment_shots)
        
        if number_of_shots == 0: 
            background_spl = None
            #append results for this frame
            results.append({"Segment_Start": current_segment_start, "Segment_End": current_segment_end,"Background Noise":background_spl})
            print("Current Segment from:",current_segment_start,"to",current_segment_end," No Shooting")
            
            #move onto next segment
            current_segment_start += pd.Timedelta(minutes = 5)
            current_segment_end += pd.Timedelta(minutes = 5)
            
        #when there IS shooting    
        else:
            
            #Get the pressures
            source_audio = segment_predictions["source_wav"].min()
            sac_file = get_sac_from_wav_name(source_audio,current_segment_start)
            
            segment_pressures,sac_start_time, sampling_rate, dt = get_segment_pressures(current_segment_start, current_segment_end, sac_file, upa_per_count)
            
            if len(segment_pressures) == 0:
                print("No pressure samples for this segment, setting sound to None")
                background_spl = None
            
            else:
                noise_est = get_noise_floor(segment_pressures,sampling_rate,n_fft=512,win_length=2,hop_length=0.1)
                
                # find the mean of all the time windows
                mean_PSD = np.mean(noise_est,axis=0)
                
                #frequencies in the PSD axis
                frequencies = np.linspace(0,sampling_rate / 2,noise_est.shape[1])
                
                #filter to 18-34Hz
                low_freq= 18
                high_freq = 34
                frequencies_filter = ((frequencies >= low_freq)& (frequencies <= high_freq))
                
                #integrate area under curve
                area = trapezoid(mean_PSD[frequencies_filter]*2, frequencies[frequencies_filter])
                background_spl = 10 * np.log10(area / (1e-6 ** 2))

            
            #append all results for this frame
            results.append({"Segment_Start": current_segment_start, "Segment_End": current_segment_end,"Background Noise":background_spl})
            
            print("Current Segment from:",current_segment_start,"to",current_segment_end," Background Noise:",background_spl)
            
            
            current_segment_start += pd.Timedelta(minutes = 5)
            current_segment_end += pd.Timedelta(minutes = 5)
        
    results = pd.DataFrame(results)
    return results




###########
#run script

shots = read_shot_file(shotpath)
for predictions_file in os.listdir(predictions_folder):
    
    if not predictions_file.lower().endswith(".csv"):
        continue

    predictions_path = predictions_folder + "\\" + predictions_file
    instrument = get_station_from_filename(predictions_path)
    print("Instrument Being Processed: ",instrument, "Path: ", predictions_path)
    
    predictions = read_predictions_file(predictions_path)

    results = process_5_min_segments(shots,predictions)
    
    ######SAVE FILES################
    results_output_path = output_folder + "\\" + instrument + "_5min_backgroundnoise.csv"
    results.to_csv(results_output_path, index=False)
