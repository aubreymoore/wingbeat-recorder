# Configuration file for wingbeat-recorder
##########################################
# USB device card number and device number (discover with `arecord -l`)
DEVICE = '2,0'
# duration of each recording in seconds
DURATION = 1
# time in seconds between the start of time of consecutive recordings
# Set to 0 for maximum processing speed
CYCLETIME = 10
# recording format (choose from: wav, dat, raw, ogg, flac; see arecord docs for details)
FORMAT = 'dat'
# number of channels (1 for mono, 2 for stereo)
CHANNELS = 1
# number of recordings to make
NRECORDINGS = 2
# directory to save the recordings as wav files
# you can use absolute (recommended) or relative paths
# can be a nested path such as '/home/aubrey/Desktop/wingbeat-recorder/data/recordings/honeybees'
WAVFILE_DIR = '/home/aubrey/Desktop/wingbeat-recorder/data/recordings'
# directory to save wingbeat waveforms as wav files
# you can use absolute (recommended) or relative paths
# can be a nested path such as '/home/aubrey/Desktop/wingbeat-recorder/data/waveforms/honeybees'
#
#########################################################################################
# The next few parameters are used for finding and extracting waveforms from the wav file
#########################################################################################
WAVEFORM_DIR = '/home/aubrey/Desktop/wingbeat-recorder/data/waveforms'
# width of the waveform in samples
WAVEFORM_WIDTH = 10000
# parameter for finding waveform peaks in a wav file recording
# this is the minimum number of standard deviations above the mean of the wav data
PEAK_THRESHOLD = 4.5
# plot wav files with peaks indicated? (True or False)
SAVE_WAVFILE_PLOTS = True
# directory for plots of wav files with peaks indicated
WAVFILE_PLOT_DIR = '/home/aubrey/Desktop/wingbeat-recorder/data/wavfile_plots'

# delete raw wav files after processing
DEL_WAV = False