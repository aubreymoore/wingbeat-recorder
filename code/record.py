# Run a recording session using parameters from config.py

from config import *
import wbrec as wbrec
import time
import os

# ensure output directories exist
os.makedirs(WAVFILE_DIR, exist_ok=True)
os.makedirs(WAVEFORM_DIR, exist_ok=True)

# ensure wavfile.csv exists
wavefile_csv_path = f'{DATA_DIR}/wavfile.csv'
if not os.path.exists(wavefile_csv_path):
    with open(wavefile_csv_path, 'w') as f:
        f.write('start_time,min,max,mean,std,num_waveforms\n') # header row

wbrec.display_config()

# run 1s test recording
wbrec.run_test_recording()
print()

# run recording session
for i in range(NRECORDINGS):
    print(i+1)
    
    # recording
    start_time = time.time()    
    wavfile_path = wbrec.record(DEVICE, DURATION, FORMAT, CHANNELS, WAVFILE_DIR)
    end_time = time.time()
    recording_time = end_time - start_time
    print(f'  recording time: {recording_time:.2f}s')

    # extract waveforms
    start_time = time.time()
    print(f'  extracting waveforms from {wavfile_path}')
    results = wbrec.extract_waveforms(wavfile_path, WAVEFORM_DIR, PEAK_THRESHOLD, WAVEFORM_WIDTH)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f'  processing time: {processing_time:.2f}s')
    
    # If DEL_WAV is True, delete the recording immediately after extracting waveforms
    if DEL_WAV:
        # delete wav file
        print(f'  deleting {wavfile_path}')
        os.remove(wavfile_path)
    
    # append a line to wavfile.csv
    with open(wavefile_csv_path, 'a') as f:
        start_time = os.path.basename(wavfile_path).replace('.wav', '')
        s = f'{start_time},{results["min"]},{results["max"]},{results["mean"]},{results["std"]},{results["num_waveforms"]}\n'
        print(f'  appending data to wavefile.csv: {s.strip()}')
        f.write(s)
    
    # sleep until time for next recording cycle
    
    if i < NRECORDINGS - 1:
        sleeptime = int(CYCLETIME - (recording_time + processing_time))
        if sleeptime >= 0:
            print(f'  sleeping for {sleeptime}s before starting next recording')
            time.sleep(sleeptime)
        
print('\nfinished')