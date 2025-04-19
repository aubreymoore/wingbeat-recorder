# Run a recording session using parameters from config.py

from config import *
import wbrec as wbrec
import time
import os

# ensure output directories exist
os.makedirs(WAVFILE_DIR, exist_ok=True)
os.makedirs(WAVEFORM_DIR, exist_ok=True)
os.makedirs(WAVFILE_PLOT_DIR, exist_ok=True)

# run 1s test recording
wbrec.display_config()
wbrec.run_test_recording()
print()

# run recording session
for i in range(NRECORDINGS):
    
    # recording
    start_time = time.time()    
    wavfile_path = wbrec.record(DEVICE, DURATION, FORMAT, CHANNELS, WAVFILE_DIR)
    end_time = time.time()
    recording_time = end_time - start_time

    # analyze
    start_time = time.time()
    wbrec.extract_waveforms(wavfile_path, WAVEFORM_DIR, SAVE_WAVFILE_PLOTS, PEAK_THRESHOLD, WAVEFORM_WIDTH, WAVFILE_PLOT_DIR)
    end_time = time.time()
    analysis_time = end_time - start_time
    
    # sleep until time for next recording cycle
    sleeptime = int(CYCLETIME - (recording_time + analysis_time))
    if sleeptime >= 0:
        print(f'  sleeping for {sleeptime}s before starting next recording')
        time.sleep(sleeptime)
    
print('\nfinished')