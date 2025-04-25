""" 
**btlflt.py**

This file contains code for analyzing images extracted from the Beetles in Flight video
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video, Audio
from icecream import ic
from scipy.io import wavfile
import wave
from scipy.signal import butter, lfilter, correlate, freqz
from scipy.fft import rfft, rfftfreq
import ffmpeg

def create_grayscale_image(image_path:str, output_path:str)->None:
    """
    Converts a color image to grayscale and saves it.
    """
    img = cv2.imread(image_path, 0)
    cv2.imwrite(output_path, img)
    
# create_grayscale_image('code/frames/0001.png', 'gray.png')


def create_square_wav(filepath:str, sample_rate:int, frequency:float, duration:float)->None:
    """ 
    Creates and saves a WAV file. Data saved as np.float32 ranging from -0.99 to 0.99
    """
    t = np.linspace(start=0., stop=duration,  num= int(duration * sample_rate), endpoint=False)
    square_wave = 0.99 * signal.square(2. * np.pi * frequency * t)
    wavfile.write(filepath, sample_rate, square_wave.astype(np.float32))
# create_square_wav(filepath='square_wave.wav', sample_rate=6000, frequency=100., duration=0.1)

def create_sine_wav(filepath:str, sample_rate:int, frequency:float, duration:float)->None:
    """ 
    Creates and saves a WAV file. Data saved as np.float32 ranging from -0.99 to 0.99
    """
    t = np.linspace(start=0, stop=duration, num=int(duration * sample_rate), endpoint=False)
    sine_wave = 0.99 * np.sin(2. * np.pi * frequency * t)
    wavfile.write(filepath, sample_rate, sine_wave.astype(np.float32))
# create_sine_wav(filepath='sine_wave.wav', sample_rate=6000, frequency=100., duration=0.1)

def convert_wav_from_float64_to_int16(src_wav_path, dest_wav_path):
    """ 
    Converts a WAV file from float64 to int16.
    This function facilitates using a WAV file in a web page with HTML code like:
    HTML example: <audio controls src="myfile.wav"></audio>
    This code does not work for 64-bit or 32-bit WAV files.
    """
    samplerate, data = wavfile.read('beetle.wav')
    if data.dtype == 'float64':
        maxint = np.iinfo(np.int16).max
        data = np.int16((maxint - 1) * data)   # rescales data values from (-1.0, 1.0) to (-32766, 32766)
        wavfile.write('beetle_16bit.wav', samplerate, data)
    else:
        print(f'WARNING: {src_wav_path} was not converted because source data.dtype is not float64')       
# convert_wav_from_float64_to_int16('beetle.wav', 'beetle_16bit.wav')

# def calc_intensity_list(images_dir, first_frame_num, last_frame_num, normalize=True):
#     """
#     Calculates the average intensity of pixels within each frame of the video scene.
#     If normalize is True, intensity_list will be normalized so that mean is zero and range is -1 to 1.
#     """
#     intensity_list = []
#     for frame_num in range(first_frame_num, last_frame_num + 1):
#         image = cv2.imread(f'{images_dir}/{frame_num:04d}.png', cv2.IMREAD_GRAYSCALE)
#         intensity = np.mean(image)
#         intensity_list.append(intensity)
#     if normalize:
#         intensity_list = normalize_array(intensity_list)
#     return intensity_list

# calc_intensity_list(FRAMES_DIR, FIRST_FRAME_NUM, LAST_FRAME_NUM)


def calc_intensity_time_series(images_dir:str, first_image:int, last_image:int)->None:
    """
    Calculates the average intensity of pixels within each frame of the video scene.
    If normalize is True, intensity_list will be normalized so that mean is zero and range is -1 to 1.
    """
    intensity_list = []
    for image_num in range(first_image, last_image + 1):
        image = cv2.imread(f'{images_dir}/{image_num:04d}.png', cv2.IMREAD_GRAYSCALE)
        intensity = np.mean(image)
        intensity_list.append(intensity)
    return intensity_list

# calc_intensity_list(FRAMES_DIR, FIRST_FRAME_NUM, LAST_FRAME_NUM)


def plot_time_frequency(samplerate, data):
    """
    Plots data in the time domain and the frequency domain.
    The figure and its 2 sets of axes are returned so that these objects can be modified.
    """
    duration = data.shape[0] / samplerate
    time = np.linspace(0., duration, data.shape[0])

    # Perform FFT and calculate frequency array
    power = np.abs(rfft(data))
    freq = rfftfreq(data.shape[0], 1/samplerate)
    
    # Create plot
    fig, (axt, axf) = plt.subplots(2, 1, constrained_layout=1, figsize=(7, 7))

    axt.plot(time, data, lw=1)
    axt.set_xlabel('time (s)')
    axt.set_ylabel('amplitude')

    axf.plot(freq, power, lw=1)
    axf.set_xlabel('frequency (Hz)')
    axf.set_ylabel('amplitude')
    axf.fill_between(freq, power)
    
    return fig, (axt, axf)


def create_frame_intensity_figure(frame_num, seconds_list, intensity_list, frame_path, fig_path):
    """
    Creates an image with two subplots: the frame image and the intensity plot.
    """
 
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6))

    # Create image plot
    image = plt.imread(frame_path)
    ax1.imshow(image)
    ax1.set_title(f'Frame {frame_num} | normalized mean pixel intensity: {intensity_list[frame_num-1]:.2f}')
    ax1.axis('off')

    # Create a sample plot
    x = seconds_list[:frame_num]
    y = intensity_list[:frame_num]

    ax2.plot(x, y)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('normalized mean pixel intensity')
    ax2.set_xlim(min(seconds_list), max(seconds_list))
    ax2.set_ylim(-1, 1)

    # plt.tight_layout()
    
    # Save figure in a file
    fig.tight_layout
    fig.savefig(fig_path)
    plt.close(fig)
    
    return fig
   
# create_frame_intensity_figure(1, seconds_list, intensity_list, 'frames/0001.png', 'test.png')

# from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()
    
# run()

def create_wav_file(filename:str, data:np.array, framerate:int=44100)->None:
    """ 
    Creates a 16-bit mono WAV file. 
      
    data is a numpy array with elements ranging between -1.0 and 1.0 inclusive
    data is rescaled to range between -32767 and +32767 stored in little-endian format
    """
    data = normalize_array(data) # normalize data to range between -0.99 and 0.99
    data = np.clip(np.round(data * 32767), -32767, 32767).astype("<h")
    with wave.open(filename, mode="wb") as wav_file:
        wav_file.setframerate(framerate)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.writeframes(data) 
        
# # create a WAV file containing 1 second of white noise sampled at 8 kHz
# data = np.random.uniform(low=-1.0, high=1.0, size=8000)
# create_wav_file(filename='white_noise.wav', data=data, framerate=8000)

# Code from https://stackoverflow.com/questions/61534687/how-to-calculate-pitch-fundamental-frequency-f-0-in-time-domain

# import numpy as np
# from scipy.io import wavfile
# from scipy.signal import correlate, fftconvolve
# from scipy.interpolate import interp1d

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def video2frames(video_path: str, images_dir: str, image_fname_pattern: str='%04d.png')->None:
    """
    Converts a video file into an image sequence.
    """
    if os.path.exists(images_dir):
        print(f'Images directory {images_dir} already exists. Skipping video2frames conversion.')
    else:
        os.makedirs(images_dir)
        # command = f'ffmpeg -r 1 -i {video_path} -r 1 {images_dir}/{image_fname_pattern}'
        # os.system(command)
        (
            ffmpeg
            .input(video_path)
            .output(f'{images_dir}/{image_fname_pattern}')
            .run()
        )

        

def frames2video(images_dir: str, video_path: str, image_fname_pattern: str='%04d.png', fps:int=30)->None:
    """
    Converts an image sequence into a video file.
    """
    if os.path.exists(video_path):
        print(f'Video file {video_path} already exists. Skipping frames2video conversion.')
    else:
        # command = f'ffmpeg -r {fps} -i {images_dir}/{image_fname_pattern} -vcodec libx264 -crf 28 {video_path}'
        # os.system(command)
        (
            ffmpeg
            .input(f'{images_dir}/{image_fname_pattern}', framerate=fps)
            .output(video_path, crf=28)
            .run()
        )
        
        
def remove_background(input_image_path:str, output_image_path:str, mask_path:str, thresh:int=140, maxval:int=255)->None: 
    """ 
    Remove the background from an image using a binary threshold and saves in output_image_path.
    The mask is saved as a binary image.
    """
    image = cv2.imread(input_image_path)

    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply a binary threshold to the image
    _, binary = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(image_rgb)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Bitwise-and the mask and the original image
    result = cv2.bitwise_and(image_rgb, mask)
    
    # Save the mask 
    cv2.imwrite(mask_path, mask)
    
    # Save the masked image
    cv2.imwrite(output_image_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def normalize_array(arr:np.array)->np.array:
    """ 
    Normalize a 1d array to range between -0.99 and 0.99 with a mean of 0.
    """ 
    
    # set mean to zero
    arr = arr - np.mean(arr)
    
    # calculate scale factor
    scale_factor = 0.99 / max(abs(np.min(arr)), np.max(arr))
    return scale_factor * arr

# for i in range(10):
#     data = np.random.uniform(low=0, high=255, size=10)
#     normalized_data = normalize_array(data)
#     print(f'{i}   {np.mean(normalized_data)=:.2f}   {np.min(normalized_data)=:.2f}   {np.max(normalized_data)=:.2f}')
