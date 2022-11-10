import glob
import os
import mne
import numpy as np

#First, investigate cassette files.
data_list = sorted(glob.glob('/DataCommon2/wypark/sleep-cassette/**'))
trains = [x for x in data_list if x.endswith('PSG.edf')]
labels= [x for x in data_list if x.endswith('Hypnogram.edf')]

print('number of train samples:',len(trains))
print('number of labels:',len(labels))


for i in range(len(trains)):
    print(i)
    subjetc_id = trains[i].split('/')[-1].split('-')[0]

    # load signal and upsampling
    data = mne.io.read_raw_edf(trains[i])
    data = data.get_data()[:2, :]
    data = mne.filter.resample(data, up=2.0)  # upsampling to 200Hz

    # load annotaion
    ann = mne.read_annotations(labels[i])
    label_list = []
    for dur, des in zip(ann.duration, ann.description):
        for i in range(int(dur) // 30):
            label_list.append(des[-1])

    n = 0
    for slice_index in range(data.shape[1] // (200 * 30)):  # 200Hz, 30epochs
        # ingore the no labels
        if label_list[slice_index] == '?':
            n = 0
            continue
        # If the wake state is repeated more than 60 times, cut off the remaining wake state.
        elif label_list[slice_index] == 'W':
            n += 1
        else:
            n = 0

        if n >= 61:
            continue

        adress = '/DataCommon2/wypark/Preprocessed_SleepEDF_expanded_remove_long_wake/'
        if not os.path.exists(adress + '/' + str(subjetc_id)):
            os.makedirs(adress + '/' + str(subjetc_id))

        data_adress = adress + '/' + str(subjetc_id)

        data_path = data_adress + '/' + str(subjetc_id) + '_' + str(slice_index) + '.npz'
        X = data[:, slice_index * 200 * 30: (slice_index + 1) * 200 * 30]
        Y = label_list[slice_index]
        np.savez(data_path, x=X, y=Y)