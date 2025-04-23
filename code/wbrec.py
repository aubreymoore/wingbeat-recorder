# Functions for recording and analyzing insect wingbeat waveforms

from config import *
import subprocess
import datetime
import os
import time
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
# import matplotlib.pyplot as plt
from icecream import ic


def record(device, duration, format, channels, wavfile_dir):
    timestamp = datetime.datetime.now().astimezone().isoformat()
    wavfile_path = f'{wavfile_dir}/{timestamp}.wav'
    command = f'arecord -D hw:{device} -d {duration} -f {format} -c {channels} {wavfile_path}'
    subprocess.run(command, shell=True)
    return wavfile_path


def extract_waveforms(wavfile_path: str, waveform_dir: str, peak_threshold: float, waveform_width: int):
    """
    Extract waveforms (transients) from a wav file and save them as separate wav files.
    """
    # print(f'  extracting waveforms from {wavfile_path}')
    
    # Calculate the height parameter for the find_peaks function
    sr, data = wavfile.read(wavfile_path)
    mean = np.mean(data)
    std = np.std(data)
    height = mean + peak_threshold * std    
    peaks, _ = find_peaks(data, height=height, distance=waveform_width)
    print(f'    found {len(peaks)} peaks')
    results = {'mean': mean, 'std': std, 'min': np.min(data), 'max': np.max(data)}
    
    num_waveforms = 0
    for peak in peaks:
        start = peak - waveform_width // 2
        stop = peak + waveform_width // 2
        if start >= 0 and stop <= len(data):
            waveform = data[start:stop]
            
            # Calculate time between the start of the wav file and the start of the transient waveform
            seconds = start/sr
            
            # Calculate the new timestamp and use this to name the wav file for the transient waveform
            timestamp_str = wavfile_path.split('/')[-1].replace('.wav', '')
              
            # Extract the timestamp from the wav file name
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f%z')
            
            # Add the seconds to the timestamp
            waveform_timestamp = timestamp + datetime.timedelta(seconds=seconds)
        
            # Create the new wav file name
            waveform_file_path = f'{waveform_dir}/{waveform_timestamp.isoformat()}.wav'
            
            # Save the transient waveform as a wav file
            wavfile.write(waveform_file_path, sr, waveform)
            
            num_waveforms += 1
    results['num_waveforms'] = num_waveforms
        
    return results
        
    # if plot_peaks:
    #     plt.plot(data)
    #     plt.plot(peaks, data[peaks], 'x')
    #     plt.plot(np.zeros_like(data), '--', color='gray')
    #     plt.savefig(f'{wavfile_plot_dir}/{os.path.basename(wavfile_path).replace(".wav", ".png")}')
    #     plt.close()

# wavfile_path = '/home/aubrey/Desktop/wingbeat-recorder/data/recordings/2025-04-22T17:29:41.225735+08:00.wav'     
# results = extract_waveforms(wavfile_path, WAVEFORM_DIR, PEAK_THRESHOLD, WAVEFORM_WIDTH)
# ic(results)


def analyze(wavfile):
    print(f'  analyzing {wavfile}')
    time.sleep(15)


def run_test_recording():
    """
    Runs a 1s test recording. An exception is raised to prevent further processing if the test fails.
    """
    print('Running a 1s test recording.')
    wavfile_path = record(device=DEVICE, duration=1, format=FORMAT, channels=CHANNELS, wavfile_dir=WAVFILE_DIR)
    if os.path.exists(wavfile_path):
        print(f"  test recording saved to {wavfile_path}")
        os.remove(wavfile_path)  # remove the test recording
        print("  test recording deleted")
        print('  test recording was a success') 
    else:
        print('\nERROR: Test recording failed.\n  Check DEVICE setting in config.py \n  Check if device is connected\n')
        
        # Get info on audio devices
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        print(result.stdout)
        
        raise Exception('Test recording failed')
    
# run_test_recording()
# print('this line should not be visible if the test recording fails')


def display_config():
    """
    Displays config.py.
    """
    print('\nParameters - see config.py for details.')
    with open('code/config.py', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith(('#', '\n')):
                print(line.strip())
    print()
            
display_config()

    