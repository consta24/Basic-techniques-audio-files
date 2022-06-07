from scipy.signal import butter, freqz, lfilter, decimate, iircomb
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d

def interp(ys, mul):
    # linear extrapolation for last (mul - 1) points
    ys = list(ys)
    ys.append(2*ys[-1] - ys[-2])
    # make interpolation function
    xs = np.arange(len(ys))
    fn = interp1d(xs, ys, kind="cubic")
    # call it on desired data points
    new_xs = np.arange(len(ys) - 1, step=1./mul)
    return fn(new_xs)



fs, data = wavfile.read('ele.wav')

plt.close('all')

plt.figure(1, figsize = (24, 9))
plt.clf()
plt.plot(data)
plt.title("Elephant song yey")

#Lowpass filter effects of a signal (vocal) with different cut-off frequencies: 6000, 3400 and 2000 Hz

#We plot 3 lowpass IIR filters of order 20, starting from Butterworth equivalents.

[b1, a1] = butter(20, 6000/(fs/2), btype = 'low')
[b2, a2] = butter(20, 3400/(fs/2), btype = 'low')
[b3, a3] = butter(20, 2000/(fs/2), btype = 'low')

[h1, w1] = freqz(b1, a1, 256, fs);
[h2, w2] = freqz(b2, a2, 256, fs);
[h3, w3] = freqz(b3, a3, 256, fs);

plt.figure(2, figsize=(24, 18))
plt.clf()

plt.subplot(3, 1, 1)
plt.title("""Lowpass Filters
fc1 = 6 kHz""")
plt.grid(True)
plt.xlabel("Frequency(Hz)")
plt.ylabel("Response module in frequency")
plt.plot(np.abs(h1), w1)

plt.subplot(3, 1, 2)
plt.title("fc2 = 3.4 kHz")
plt.grid(True)
plt.xlabel("Frequency(Hz)")
plt.ylabel("Response module in frequency")
plt.plot(np.abs(h2), w2)

plt.subplot(3, 1, 3)
plt.title("fc3 = 2 kHz")
plt.grid(True)
plt.xlabel("Frequency(Hz)")
plt.ylabel("Response module in frequency")
plt.plot(np.abs(h3), w3)

plt.show()

#Filtering of the audio signal

y1 = lfilter(b1, a1, data);
y2 = lfilter(b2, a2, data);
y3 = lfilter(b3, a3, data);

n01 = data.size / fs # duration
n02 = np.arange(0, data.size) / fs

plt.figure(3, figsize=(24, 18))
plt.clf()

plt.subplot(3, 1, 1)
plt.plot(n02, y1)
plt.title('Filtered signal with three lowpass filters: fc = 6 kHz, 3.4 kHz, 2 kHz ')

plt.subplot(3, 1, 2)
plt.plot(n02, y2)

plt.subplot(3, 1, 3)
plt.plot(n02, y3)

plt.xlabel("Time(s)")

wavfile.write("y1.wav", fs, y3.astype('int16'))
wavfile.write("y2.wav", fs, y3.astype('int16'))
wavfile.write("y3.wav", fs, y3.astype('int16'))

yd = decimate(data, 2)
yi = interp(data, 2)

wavfile.write("yd.wav", fs, yd.astype('int16'))
wavfile.write("yi.wav", fs, yi.astype('int16'))

# Visualization of signals with different playback speed

fs1, data1 = wavfile.read('yd.wav')
fs2, data2 = wavfile.read('yi.wav')

n11 = data1.size / fs1 # duration
n12 = np.arange(0, data1.size) / fs1

n21 = data2.size / fs2 # duration
n22 = np.arange(0, data2.size) / fs2

plt.figure(4, figsize=(24, 18))
plt.clf()

plt.subplot(2, 1, 1)
plt.plot(n12, data1)

plt.title('Initial signal decimated and interpolated, with different playback speed.')

plt.subplot(2, 1, 2)
plt.plot(n22, data2)

plt.xlabel("Time(s)")

#Simulation of a reverb effect / echo

Q = 30 # quality factor

bnotch, anotch = iircomb(fs/(fs/2), Q, ftype='notch', fs=fs)

ynotch = lfilter(bnotch, anotch, data)
hnotch, wnotch = freqz(bnotch, anotch, 256, fs)

plt.figure(5, figsize=(24, 18))
plt.clf()


plt.subplot(2, 1, 1)
plt.plot(abs(hnotch), wnotch)
plt.title("IIRCOMB NOTCH")
plt.subplot(2, 1, 2)
plt.plot(n02, ynotch)

wavfile.write("ynotch.wav", fs, ynotch.astype('int16'))

bpeak, apeak = iircomb(fs/(fs/2), Q, ftype='peak', fs=fs)

ypeak = lfilter(bpeak, apeak, data)
hpeak, wpeak = freqz(bpeak, apeak, 256, fs)

plt.figure(6, figsize=(24, 18))
plt.clf()

plt.subplot(2, 1, 1)
plt.plot(abs(hpeak), wpeak)
plt.title("IIRCOMB PEAK")
plt.subplot(2, 1, 2)
plt.plot(n02, ypeak)

wavfile.write("ypeak.wav", fs, ypeak.astype('int16'))











