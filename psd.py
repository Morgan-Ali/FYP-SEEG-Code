import sys
import getpass
import numpy as np
import hdf5storage

# MNE is a library purpose built for analysing neurological activity data such as EEG
import mne

# Implements Power Spectral Density on SEEG data for feature extraction.
# Adapted from getInput_psd.py by Xiaolong Wu.

# Subject's unique numeric identifier, e.g. P32
subject_id = 32

# Sampling frequency of the SEEG data
fs = 1000

# For windowing Epochs, decide size of window and distance to shift along
window_size = 500
window_step = 250


def window_epochs(epoch, window_size, window_step):
    """Epochs must be properly windowed for use in the EpochArray object

    Input Arguments:
    
    epoch   -   List of epochs after an event
    size    -   Integer size of window
    step    -   Integer size of shift along the epoch each iteration

    Output Arguments

    trials_out  -   List of windowed epochs
    """

    epoch_length = epoch.shape[2]
    trials_out = []
    trials_list = []

    for trial in epoch:

        i = 0

        # Add windows to trial list until we run out of data in the epoch
        while window_step * i + window_size < epoch_length:

            start = i * window_step
            trials_list.append(trial[:, start : (start + window_size)])
            i += 1

        # Add a final window if needed
        i -= 1
        if window_step * i + window_size < epoch_length - 100:
            trials_list.append(trial[:, -window_size:])

    trials_out.append(trials_list)
    trials_out = np.concatenate(trials_out, axis=0)  # (1300, 63, 500)

    return trials_out


# Get device's current user name
user = getpass.getuser()

# Import SEEG data and find total number of useful channels
data_path = "C:/Users/"+user+"/Downloads/share_8_good/gesture/preprocessing/P"+str(subject_id)+"/preprocessing2.mat"
matlab_data = hdf5storage.loadmat(data_path)
data = matlab_data['Datacell']
good_channels = matlab_data['good_channels']
total_good_channels = len(np.squeeze(good_channels))
data = np.concatenate((data[0,0],data[0,1]),0)

print("Data from subject P"+str(subject_id)+" loaded")
print(str(total_good_channels)+" total good channels")

# Information tags about the purpose of each channel
seeg_tags = ["seeg"] * total_good_channels
names = np.array(seeg_tags + ["emg0", "emg1", "stim_trigger", "stim_emg"])
types = np.array(seeg_tags + ["emg", "emg", "stim", "stim"])

# Create an instance of MNE Raw to represent the SEEG data
info = mne.create_info(ch_names=list(names), ch_types=list(types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)

print("Instance of MNE Raw created")

# Locate where stimulus for gesture events were applied
events_stim_emg = mne.find_events(raw, stim_channel='stim_emg')
events_stim_emg = events_stim_emg - [0,0,1]


# Create Epochs 4 seconds before and 4 seconds after the movement event
# 8 seconds total per trial with 20 total trials per epoch
epochs = mne.Epochs(raw.pick(["seeg"]), events_stim_emg, tmin=-3, tmax=4,baseline=None)
epochs = epochs.load_data().pick(picks="all") 
epoch_info = epochs.info

# Baseline - 4 seconds before
baseline_epochs = [epochs['0'].load_data().copy().crop(-3,0),
                   epochs['0'].load_data().copy().crop(-3,0),
                   epochs['0'].load_data().copy().crop(-3,0),
                   epochs['0'].load_data().copy().crop(-3,0),
                   epochs['0'].load_data().copy().crop(-3,0)]

# 4 seconds after
epoch_list = [epochs['0'].load_data().crop(0,4).get_data(),
              epochs['1'].load_data().crop(0,4).get_data(),
              epochs['2'].load_data().crop(0,4).get_data(),
              epochs['3'].load_data().crop(0,4).get_data(),
              epochs['4'].load_data().crop(0,4).get_data()]

print("Epochs Finalised")

# Windowing epochs for each gesture
for i in range(5):

    # Create windowed epochs and create an instance of MNE Epochs object
    windowed_epoch = window_epochs(epoch_list[i],window_size, window_step)
    temp_epoch_array = mne.EpochsArray(windowed_epoch,epoch_info)

    # Adjust numerical identifier for events
    events = temp_epoch_array.events
    events[:,2] = i+1

    # Replace events and move Epochs object to list
    temp_epoch_array.events = events
    epoch_list[i] = temp_epoch_array

# Perform PSD:

number_of_trials = windowed_epoch.shape[0] 
epoch_list_psd = []
baseline_epochs_psd = []

# Perform PSD for the baseline epochs
for i in range(5):

    # Use function for Welch's PSD estimate
    (temp,freq) = mne.time_frequency.psd.psd_welch(baseline_epochs[i])

    # Calculate PSD average
    temp = np.mean(temp,axis=0)
    baseline_epochs_psd.append(temp)

print("PSD for baseline epochs complete")

# Perform PSD for event epochs
for i in range(5):

    (temp,freq) = mne.time_frequency.psd.psd_welch(epoch_list[i])
    epoch_list_psd.append(temp)
    
print("PSD for event epochs complete")


# Averaging across frequency bands:


frequency_bands = [[1,4],[4,8],[8,13],[13,30],[60,75],[75,95],[105,125],[125,145],[155,195]]
channel_n = epoch_list_psd[0].shape[1]
number_of_bands = len(frequency_bands)

baseline_psd_average = []

for i in range(5):

    baseline_psd_temp = np.zeros([channel_n, number_of_bands])

    for k, band in enumerate(frequency_bands):

        # Lower and upper frequency in a given band, e.g. in (1-4Hz), 1 is lower, 4 is upper
        lower_f = band[0]
        upper_f = band[1]

        lowf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - lower_f))
        highf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - upper_f))

        baseline_psd_temp[:,k] = np.mean(baseline_epochs_psd[i][:,lowf_index:highf_index],axis=1)
    baseline_psd_average.append(baseline_psd_temp)

event_psd_average = []

for i in range(5):

    event_psd_temp = np.zeros([number_of_trials, channel_n, number_of_bands])

    for k,band in enumerate(frequency_bands):

        lower_f = band[0]
        upper_f = band[1]

        lowf_index,_=min(enumerate(freq), key=lambda x: abs(x[1] - lower_f))
        highf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - upper_f))

        event_psd_temp[:,:,k]=np.mean(epoch_list_psd[i][:,:,lowf_index:highf_index],axis=2)

    event_psd_temp = event_psd_temp - baseline_psd_average[i]
    temp = np.reshape(event_psd_temp,(number_of_trials, channel_n*number_of_bands))
    event_psd_average.append(temp)

event_psd_average = np.concatenate(event_psd_average,axis=0) # (1500, 23*9=207)

print("Average done/")

labels = []

for i in range(5):
    number_of_trials = temp.shape[0]
    label = [[i,] * number_of_trials]
    labels.append(label)

labels = np.squeeze(np.asarray(labels))
labels = np.squeeze(labels.reshape((1,-1))) # (1500,)


# Export PSD Features
filename="C:/Users/"+user+"/Downloads/share_8_good/gesture/preprocessing/P"+str(subject_id)+"/psd_feature"
#filename="C:/Users/"+user+"/Downloads/share_8_good/gesture/psd_feature"
np.save(filename, event_psd_average)
print("PSD Features exported")

# Export Feature Labels
filename="C:/Users/"+user+"/Downloads/share_8_good/gesture/preprocessing/P"+str(subject_id)+"/label"
#filename="C:/Users/"+user+"/Downloads/share_8_good/gesture/label"
np.save(filename, labels)
print("Feature Labels exported")




