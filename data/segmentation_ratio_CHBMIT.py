import glob
import re
import mne
import numpy as np
from pyedflib import highlevel
import os

file_path = '/DataCommon2/wypark/smj_data/Raw_CHB-MIT/chb-mit-scalp-eeg-database-1.0.0/'
file_list = sorted(glob.glob((file_path + '**')))
print('Total subjects:',len(file_list))

def save_the_seizrure_block():
    seizure_file = []
    seizure_time = []
    for num in range(1, 25): # subject 1~24
        file = "chb" + str(num).rjust(2, '0')
        file_path = '/DataCommon2/wypark/smj_data/Raw_CHB-MIT/chb-mit-scalp-eeg-database-1.0.0/'
        with open(file_path + file + "/" + file + "-summary.txt") as txt:
            text = txt.readlines()
            for content in text:
                if file in content:
                    edf_file_name = content.split(" ")[-1].replace("\n", "")
                    seizure_file.append(edf_file_name)

                # count seizures
                if "Number of Seizures" in content:
                    seizure_cnt = int(content[-2])
                    if seizure_cnt == 0:
                        seizure_file.pop()
                    else:
                        for _ in range(seizure_cnt - 1):
                            seizure_file.append(edf_file_name)

                start_re = re.compile('Seizure.+Start Time')
                end_re = re.compile('Seizure.+End Time')

                if start_re.match(content):
                    start_time = int(re.findall("\d+", content.split(":")[-1])[0])

                if end_re.match(content):
                    end_time = int(re.findall("\d+", content.split(":")[-1])[0])
                    seizure_time.append((start_time, end_time))
                    # except chb12 - 27, 28, 29
                    if 'chb12_27' in edf_file_name or 'chb12_28' in edf_file_name or 'chb12_29' in edf_file_name or 'chb13_40.edf' in edf_file_name or 'chb16_18.edf' in edf_file_name:
                        seizure_file.pop()
                        seizure_time.pop()

    seizure_block = []
    for i in range(len(seizure_file)):
        seizure_block.append((seizure_file[i], seizure_time[i]))

    return seizure_block


seizure_block = save_the_seizrure_block()
print(seizure_block)
# dictionary = {seizure_block[i][0] : seizure_block[i][1] for i in range(len(seizure_block))}
# for i in range(len(seizure_block)):
#     dictionary = dict.fromkeys(seizure_block[i][0],seizure_block[i][1])

# print(dictionary)
# for block in seizure_block:
#     print(block)
channel_list = ['C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8','T8-P8', 'FP1-F3', 'FP1-F7', 'FP2-F4', 'FP2-F8', 'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P8-O2', 'T7-P7','T7-FT9','FT9-FT10','FT10-T8','P7-T7']

adress = '/DataCommon2/wypark/smj_data/same_ratio_CHB_MIT_2s'
#

T = 2
R = 1

for i in range(len(file_list)):
    data_list = glob.glob(file_list[i] + '/**') # sbject's files ( edf, tet, edf.seizure)
    subj_n = file_list[i].split(('/'))[-1]
    seizure_files = [x for x in data_list if x.endswith('.seizures')]
    # print(seizure_edf_files)
    print(len(seizure_files))
    seizure_edf_files = []
    if not os.path.exists(adress + '/' + str(subj_n)):
        os.makedirs(adress + '/' + str(subj_n))

    for j in seizure_files:
        # print(len(j))
        # print(j[:-9])
        seizure_edf_files.append((j[:-9]))


    for j in range(len(seizure_edf_files)):
        seizure = seizure_edf_files[j].split('/')[-1] # seizure file name ex)chb01_15.edf
        print(seizure)
        if seizure == 'chb12_27.edf'  or seizure == 'chb12_28.edf' or seizure == 'chb12_29.edf' or seizure == 'chb13_40.edf' or seizure == 'chb16_18.edf' :
            print('f{} pass'.format(seizure))
            continue
        EEG_data = highlevel.read_edf(seizure_edf_files[j])[0] # data
        signal_headers = highlevel.read_edf(seizure_edf_files[j])[1] # label
        print('number of channels:', len(signal_headers))
        data = []
        for k in range(len(signal_headers)):
            if signal_headers[k]['label'] in channel_list :
                data.append(EEG_data[k])

        if not os.path.exists(adress + '/' + str(subj_n) +'/' + seizure.split('.')[0]):
            os.makedirs(adress + '/' + str(subj_n) +'/' + seizure.split('.')[0])

        data_adress = adress + '/' + str(subj_n) +'/' + seizure.split('.')[0]
        data =  np.array(data) # we use only 23 channels
        data = mne.filter.resample(data, down=1.28)  # downsampling to 200Hz
        # data = mne.filter.filter_data(data, sfreq=200, l_freq=0., h_freq=75.)

        seizure_time = []
        for block in seizure_block:
            if block[0] == seizure:
                seizure_time.append(block[1])
        print(seizure_time)

        seizure_dur = 0
        for time in seizure_time:
            dur = time[1] - time[0]
            seizure_dur += dur
            if dur/T == int(dur/T):
                for s in range(0, int(dur / T)):
                    data_path = data_adress + '/' + seizure.split('.')[0] + '_' + str(time[0])[:3] + str(s) + '.npz'
                    X = data[:,200*time[0] + 200*(s*T) : 200*time[0] + 200*(s+1)*T]
                    Y = 1  # seizure = 1
                    np.savez(data_path, x=X, y=Y)
            else:
                for s in range(0,int(dur/T)+1):
                    data_path = data_adress + '/' + seizure.split('.')[0] + '_' + str(time[0])[:3] + str(s) + '.npz'
                    X = data[:,200*time[0] + 200*(s*T) : 200*time[0] + 200*(s+1)*T]
                    Y = 1  # seizure = 1
                    np.savez(data_path, x=X, y=Y)
        seizure_dur = int(seizure_dur/T)*R
        while seizure_dur != 0:
            data_path = data_adress + '/' + seizure.split('.')[0] + '_nonseizure' + str(seizure_dur) + '.npz'
            n = np.random.randint(data.shape[1]-T*200)
            criterion = True
            for time in seizure_time:
                if time[0]*200 <= n and n <= time[1]*200:
                    criterion = False
            if criterion:
                X = data[:, n: n + T*200]
                Y = 0  # non_seizure = 0
                np.savez(data_path, x=X, y=Y)
                seizure_dur -= 1

