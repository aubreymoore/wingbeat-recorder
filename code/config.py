# Configuration file for wingbeat-recorder

# USB device card number and device number (discover with `arecord -l`)
DEVICE = '2,0'

# duration of each recording in seconds
DURATION = 10

# time in seconds between the start of time of consecutive recordings
# Set to 0 for maximum processing speed; a new recording will start as soon as processing of the previous one is complete
CYCLETIME = 0

# recording format (choose from: wav, dat, raw, ogg, flac; see arecord docs for details)
FORMAT = 'dat'

# number of channels (1 for mono, 2 for stereo)
CHANNELS = 1

# number of recordings to make
NRECORDINGS = 3

# directory to save output data
# you can use absolute (recommended) or relative paths
# can be a nested path such as '/home/aubrey/Desktop/wingbeat-recorder/data/honeybees'
DATA_DIR = '/home/aubrey/Desktop/wingbeat-recorder/data'

# directory to save recordings as wav files
# you can use absolute (recommended) or relative paths
# can be a nested path such as '/home/aubrey/Desktop/wingbeat-recorder/data/waveforms/honeybees'
WAVFILE_DIR = f'{DATA_DIR}/recordings'

# directory to save wingbeat waveforms as wav files
# you can use absolute (recommended) or relative paths
# can be a nested path such as '/home/aubrey/Desktop/wingbeat-recorder/data/waveforms/honeybees'
WAVEFORM_DIR = f'{DATA_DIR}/waveforms'

# The next few parameters are used for finding and extracting waveforms from the wav files
# For details, see documentation for scipy.signal.find_peaks

# width of the waveform in samples
WAVEFORM_WIDTH = 10000

# parameter for finding waveform peaks in a wav file recording
# this is the minimum number of standard deviations above the mean of the wav data

PEAK_THRESHOLD = 4.5
# plot wav files with peaks indicated? (True or False)

# delete raw wav file from WAVFILE_DIR after extracting waveforms?
DEL_WAV = True