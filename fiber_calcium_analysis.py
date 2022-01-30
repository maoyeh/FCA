import pandas as pd
import math
import numpy as np
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
import pylab as plt
import seaborn as sns
import json


class read_file:
    def __init__(self, path='', data_file='', video_align=False,
                 time_file='', ttl='', pre_event=15, post_event=25):
        self.path = path
        self.data_file = data_file
        self.video_align = video_align
        self.time_file = time_file
        self.ttl = ttl
        self.pre_event = pre_event
        self.post_event = post_event

        if self.data_file:
            self.get_data()
            self.time_tag()
            self.cut()
        else:
            print('No file selected')
            exit()

    def get_data(self):
        if self.path:
            data = pd.read_csv(self.path + '\\' + self.data_file + '.csv', skipinitialspace=True)
            with open(self.path + '\\' + self.data_file + '.json', 'r') as f:
                data_info = json.loads(f.read())
                self.sample_rate = data_info['sampling_rate']
        else:
            data = pd.read_csv(self.data_file + '.csv', skipinitialspace=True)
            with open(self.data_file + '.json', 'r') as f:
                data_info = json.loads(f.read())
                self.sample_rate = data_info['sampling_rate']

        self.Analog1 = data['Analog1'].to_numpy()
        self.Analog2 = data['Analog2'].to_numpy()
        self.Digital1 = data['Digital1'].to_numpy()
        self.Digital2 = data['Digital2'].to_numpy()

        if self.video_align:
            tag1 = self.Digital1.sum()
            tag2 = self.Digital2.sum()
            if tag1 > tag2:
                start = np.where(self.Digital1 == 1)[0][0]
                self.Analog1 = self.Analog1[start:-1]
                self.Analog2 = self.Analog2[start:-1]
                self.Digital2 = self.Digital2[start:-1]
                self.Digital1 = self.Digital1[start:-1]
                self.video_ref = 'Digital1'
            elif tag1 < tag2:
                start = np.where(self.Digital2 == 1)[0][0]
                self.Analog1 = self.Analog1[start:-1]
                self.Analog2 = self.Analog2[start:-1]
                self.Digital1 = self.Digital1[start:-1]
                self.Digital2 = self.Digital2[start:-1]
                self.video_ref = 'Digital2'

    def time_tag(self):
        if self.time_file:
            if self.path:
                time_file = pd.read_excel(self.path + '\\' + self.time_file + '.xlsx', index_col=0)
   #             time_file = pd.read_excel(self.path + '\\' + self.time_file + '.xlsx')
            else:
                time_file = pd.read_excel(self.time_file + '.xlsx', skipinitialspace=True)

            time_info = time_file.iloc[2].dropna()
            self.time_info = []
            try:
                for i in time_info:
                    t = str(i).split(':')
                    self.time_info.append(math.floor((int(t[0]) * 60 + float(t[1])) * self.sample_rate))
            except:
                print(self.path + '\\' + self.time_file + '.xlsx')
                return

        if self.ttl:
            if self.ttl == 'Digital1':
                time_info = self.Digital1
            elif self.ttl == 'Digital2':
                time_info = self.Digital2
            start_time = np.where((time_info[1:] - time_info[:-1]) == 1)[0]
            stop_time      = np.where((time_info[1:] - time_info[:-1]) == -1)[0]

            duration = stop_time - start_time
            reward = []
            r = -200

            for i in range(len(duration)):
                if start_time[i] < ( r + self.sample_rate * 60 ):
                    continue

                poke_time = duration[i]
                if poke_time > 0.1 * self.sample_rate:
                    r = start_time[i]
                    reward.append(r)

            self.time_info = reward

    def cut(self):
        self.calcium = []
        self.reference = []
        #        self.calcium = np.array([])
        #        self.reference = np.array([])
        for i in self.time_info:
            min = i - self.pre_event * self.sample_rate - 1
            max = i + self.post_event * self.sample_rate

            if min > 0 and max < len(self.Analog1):
                self.calcium.append(self.Analog1[min:max])
                self.reference.append(self.Analog2[min:max])

    def save_data(self):
        if self.path:
            txt_file = self.path + '\\' + self.data_file + '.txt'
        else:
            txt_file = self.data_file + '.txt'

        with open(txt_file, 'w') as f:
            f.write('File: ' + self.path + '\\' + self.data_file + '.csv')
            f.write('\nSampling Rate: ')
            f.write(str(self.sample_rate))
            f.write('\nBaseline(s): ' + str(self.pre_event))
            f.write('\nPost(s): ' + str(self.post_event))
            f.write('\nEvent Time Info (Data Point): \n')
            for t in self.time_info:
                f.write(str(t) + ',')
            f.write('e\n')
            f.write('\n\n\nAnalog1:\n')
            for x in self.calcium:
                for y in x:
                    f.write(str(y) + ',')
                f.write('\n')
            f.write('e\n')
            f.write('\n\n\nAnalog2:\n')
            for x in self.reference:
                for y in x:
                    f.write(str(y) + ',')
                f.write('\n')
            f.write('e')


def denoise(calcium, reference, sampling_rate=100):
    Analog1 = np.array(calcium)
    Analog2 = np.array(reference)
    calcium_denoise = []
    reference_denoise = []

    for i in range(Analog1.shape[0]):
        # Median filtering to remove electrical artifact.
       # denoised1 = medfilt(Analog1[i, :], kernel_size=5)
       # denoised2 = medfilt(Analog2[i, :], kernel_size=5)

        # Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
        b, a = butter(2, 12, btype='low', fs=sampling_rate)
        denoised1 = filtfilt(b, a, Analog1[i, :])
        denoised2 = filtfilt(b, a, Analog2[i, :])

        # Photobleaching
        b, a = butter(2, 0.001, btype='high', fs=sampling_rate)
        denoised1 = filtfilt(b, a, denoised1, padtype='even')
        denoised2 = filtfilt(b, a, denoised2, padtype='even')

        calcium_denoise.append(denoised1)
        reference_denoise.append(denoised2)

    return calcium_denoise, reference_denoise


def dF_calculate(calcium, reference, sampling_rate=100, baseline=15):
    Analog1 = np.array(calcium)
    Analog2 = np.array(reference)
    dF = []

    for i in range(Analog1.shape[0]):
        slope, intercept, r_value, p_value, std_err = linregress(x=Analog2[i, 0:(sampling_rate * baseline)],
                                                                 y=Analog1[i, 0:(sampling_rate * baseline)])
        movement = Analog2[i, :] * slope + intercept
        dF_temp = Analog1[i, :] - movement
        dF.append(dF_temp)

    return dF


def baseline_z_score(dF, sampling_rate=100, baseline=15):
    dF = np.array(dF)
    z_score = []
    for i in range(dF.shape[0]):
        baseline_mean = np.mean(dF[i, 0:(sampling_rate * baseline)])
        baseline_sd = np.std(dF[i, 0:(sampling_rate * baseline)])
        z_score_temp = (dF[i, :] - baseline_mean) / baseline_sd
        z_score.append(z_score_temp)
    return z_score


def whole_z_score(dF):
    dF = np.array(dF)
    z_score = []
    for i in range(dF.shape[0]):
        whole_mean = np.mean(dF[i, :])
        whole_sd = np.std(dF[i, :])
        z_score_temp = (dF[i, :] - whole_mean) / whole_sd
        z_score.append(z_score_temp)
    return z_score


if __name__ == '__main__':
    file_path = r'F:\Shared\File Transfers\Mao\20211221 Fear Conditionnig Fiber photometry 1st\Calcium'
    name = '291 r-2021-12-21-110400'
    test = read_file(path=file_path, data_file=name, ttl='Digital2', pre_event=5, post_event=20)

    calcium_denoised, reference_denoise = denoise(test.calcium, test.reference)
    dF = dF_calculate(calcium_denoised, reference_denoise)
    # z_score = baseline_z_score(dF,test.sample_rate[0],test.pre_event)
    z_score = whole_z_score(dF)

    time = np.linspace(0, test.pre_event + test.post_event, len(test.calcium[1]), endpoint=True)

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(9, 6)

    z_score = np.array(z_score)

    for i in range(z_score.shape[0]):
        axes[0].plot(time, test.calcium[i], color='C' + str(i), label='Channel1  ' + str(i))
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Fluorescence')
        axes[0].legend(loc='upper right', shadow=False, fontsize='x-large')

        axes[1].plot(time, test.reference[i], color='C' + str(i), label='Channel2  ' + str(i))
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Fluorescence')
        axes[1].legend(loc='upper right', shadow=False, fontsize='x-large')

    fig = plt.figure()
    axes = fig.subplots()

    if test.path:
        csv_file = test.path + '\\' + test.data_file + '_z_score.csv'
        eps_file = test.path + '\\' + test.data_file + '_z_score.eps'
        png_file = test.path + '\\' + test.data_file + '_z_score.png'
        ave_file = test.path + '\\' + test.data_file + 'ave_z_score.png'
    else:
        csv_file = test.data_file + '_z_score.csv'
        eps_file = test.data_file + '_z_score.eps'
        png_file = test.data_file + '_z_score.png'

    for i in range(z_score.shape[0]):
        axes.plot(time, z_score[i, :], color='C' + str(i), label='Tone ' + str(i))
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Fluorescence')
        axes.legend(loc='upper right', shadow=False, fontsize='x-large')

    fig = plt.figure()
    axes = fig.subplots()
    sns.heatmap(z_score, yticklabels=False, xticklabels=False, cmap='hot', vmin=-1, vmax=3)
    plt.savefig(eps_file)
    plt.savefig(png_file)

    fig = plt.figure()
    axes = fig.subplots()
    axes.plot(time, np.mean(z_score, axis=0), color='C1')

    plt.savefig(ave_file)

    z_score = pd.DataFrame(z_score)
    z_score.to_csv(csv_file)

    plt.show()
