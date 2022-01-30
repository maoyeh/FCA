import os
import fiber_calcium_analysis as fp
import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
from numpy import trapz

data_path = input('Input data path:   ')
if not data_path:
    data_path = os.getcwd()

dir_ind = 000
new_dir_name = 'analysis_00' + str(dir_ind)
new_dir = os.path.join(data_path, new_dir_name)
while os.path.isdir(new_dir):
    dir_ind += 1
    new_dir_name = 'analysis_00' + str(dir_ind)
    new_dir = os.path.join(data_path, new_dir_name)

os.mkdir(new_dir)

data = []

for f in os.listdir(data_path):
    if f.endswith('.csv'):
        if '_time' not in f:
            data_file = os.path.splitext(f)[0]
            baseline = 5 * 60
            post     = 25 * 60

          #  test = fp.read_file(path=data_path, data_file=data_file, ttl='Digital2', pre_event=baseline, post_event=post)
            test = fp.read_file(path=data_path, data_file=data_file,  video_align= True, time_file= data_file +'_time', pre_event=baseline, post_event=post)
            #calcium_denoised, reference_denoise = fp.denoise(test.calcium, test.reference)
           # dF = fp.dF_calculate(calcium_denoised, reference_denoise)
            dF = fp.dF_calculate(test.calcium,  test.reference)
            z_score = fp.baseline_z_score(dF,test.sample_rate,test.pre_event)
           # z_score = fp.whole_z_score(dF)

            time = np.linspace(0, test.pre_event + test.post_event, len(test.calcium[0]), endpoint=True)

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

            csv_file = new_dir + '\\' + data_file + '_z_score.csv'
            eps_file = new_dir + '\\' + data_file + '_z_score.eps'
            png_file = new_dir + '\\' + data_file + '_z_score.png'
            ave_file = new_dir + '\\' + data_file + 'ave_z_score.png'

            for i in range(z_score.shape[0]):
                axes.plot(time, z_score[i, :], color='C' + str(i), label='Tone ' + str(i))
                axes.set_xlabel('Time (s)')
                axes.set_ylabel('Fluorescence')
                axes.legend(loc='upper right', shadow=False, fontsize='x-large')

            fig = plt.figure()
            axes = fig.subplots()
            #sns.heatmap(z_score, yticklabels=False, xticklabels=False, cmap='hot', vmin=-1, vmax=3)
            sns.heatmap(z_score, yticklabels=False, xticklabels=False, cmap='hot')
            plt.savefig(eps_file)
            plt.savefig(png_file)

            fig = plt.figure()
            axes = fig.subplots()
            axes.plot(time, np.mean(z_score, axis=0), color='C1')

            area_uder_curve = trapz(np.mean(z_score, axis=0)[(baseline+ 300)*test.sample_rate:(baseline + 1200) * test.sample_rate], dx =  1/test.sample_rate)
            txt_file = new_dir + '\\' + data_file + '_area.txt'

            with open(txt_file, 'w') as f:
                f.write('File: ' + test.path + '\\' + test.data_file + '.csv')
                f.write('\nSampling Rate: ')
                f.write(str(test.sample_rate))
                f.write('\nBaseline(s): ' + str(test.pre_event))
                f.write('\nPost(s): ' + str(test.post_event))
                f.write('\nArea under curve:  ' + str(area_uder_curve))
          #  axes.set_ylim([-1, 3])

            plt.savefig(ave_file)

            z_score = pd.DataFrame(z_score)
            z_score.to_csv(csv_file)

            plt.close('all')

            data.append(z_score)


data = pd.DataFrame(data)
data.to_csv(new_dir + '\\'  + 'all_z_score.csv')
