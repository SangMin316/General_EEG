import glob
import os
import numpy as np
import mne
from pyedflib import highlevel


data_list = sorted(glob.glob('/DataCommon/jphyo/Dataset/MASS/SS3/SS3_EDF/**'))
trains = [x for x in data_list if x.endswith('PSG.edf')]
labels= [x for x in data_list if x.endswith('Base.edf')]

print('number of train samples:',len(trains))
print('number of labels:',len(labels))


for i in range(len(trains)):
    print(i)

    # choose EEG channels
    exclude_channels = ['EEG A2-LER'] # we use 20channel, MASS SS3 has two type channel, one has more 1 channle EEG A2-LER 22
    signal_headers = highlevel.read_edf(trains[i])[1]

    for j in range(len(signal_headers)):
        if signal_headers[j]['label'].split(' ')[0] != 'EEG':
            exclude_channels.append(signal_headers[j]['label'])

    data = mne.io.read_raw_edf(trains[i], exclude=exclude_channels)
    data = data.get_data()
    data = mne.filter.resample(data, down=1.28)  # downsampling to 200Hz
    data = mne.filter.filter_data(data, sfreq=200, l_freq=0., h_freq=75.)

    # load annotaion
    ann = mne.read_annotations(labels[i])

    for slice_index in range(len(ann.description)):  # 200Hz, 30epochs
        # ingore the no labels
        if ann.description[slice_index] == 'Sleep stage ?':
            continue

        adress = '/DataCommon2/wypark/smj_data/Preprocessed_MASS3_EDF_ch20'

        if not os.path.exists(adress + '/SS3_{}'.format(i)):
            os.makedirs(adress + '/SS3_{}'.format(i))

        data_adress = adress + '/SS3_{}'.format(i)

        data_path = data_adress + '/' + str(i) + '_' + str(slice_index) + '.npz'
        X = data[:, slice_index * 200 * 30: (slice_index + 1) * 200 * 30]
        Y = ann.description[slice_index]
        np.savez(data_path, x=X, y=Y)