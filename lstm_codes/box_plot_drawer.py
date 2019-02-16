import matplotlib.pyplot as plt
import collections
import numpy as np
import os


data_pozyx = [[1,2,5], [5,7,2,2,5], [7,2,5]]
data_RO_SLAM = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
data_MLP = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
data_BI_LSTM = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
data_OURS = [range(1,10), [1,2,5,3,2], [2,3,5,1]]

ticks = ['3', '5', '8']

class BoxPlotModule:
    def __init__(self):
        self.tmp_dict = None
        self.gt_dir_path = None
        self.dict_gt = None
        
        self.plt = plt
        
        self.target_dir_path = {}
        self.dict_al = collections.OrderedDict()
        
        self.dict_error = collections.OrderedDict()
        
        self.list_al_name = None
        self.ticks = []
        
        self.color_set = ['#34B291', '#FFC000', '#EE3030', '#30EE30', '#3030EE']
        
    def do_test(self, gt_path, al_path_list):
        self.load_gt_set(gt_path)
        for al_path in al_path_list:
            self.load_al_set(al_path, os.path.split(al_path)[1])
        self.get_dict_error()
        self.do()
        
    def load_gt_set(self, dir_path):
        self.gt_dir_path = dir_path
        self._read_data_from_gt_(dir_path)

    def load_al_set(self, dir_path, al_name):
        self.target_dir_path[al_name] = dir_path
        self.dict_al[al_name] = self._read_data_from_set_(dir_path)
        self.list_al_name = list(self.dict_al)
        
    def get_dict_error(self):
        for al_name in self.dict_al.keys():
            self._cal_error_per_al_(al_name)
            
    def _cal_error_per_al_(self, al_name):
        self.dict_error[al_name] = []
        for item in self.ticks:
            self.dict_error[al_name].append(np.sqrt(np.sum(np.square(self.dict_al[al_name][item] - self.np_gt), axis=1)).tolist())
            
    def _read_data_from_gt_(self, input_path):
        self.np_gt = np.loadtxt(input_path, delimiter=',')[-4800:,-2:]
        
        
    def _read_data_from_set_(self, input_path):
        tmp_dict = collections.OrderedDict()
        file_list = os.listdir(input_path)
        for file_name in file_list:
            file_path = os.path.join(input_path, file_name)
            if os.path.isdir(file_path):
                continue
            item_name = os.path.split(file_name)[1].split('.')[0].split('_')[-1]
            tmp_dict[item_name] = np.loadtxt(file_path, delimiter=',')[-4800:,:]
        self.ticks = list(tmp_dict.keys())
        return tmp_dict

    def _set_box_color_(self, bp, color):
        self.plt.setp(bp['boxes'], color=color)
        self.plt.setp(bp['whiskers'], color=color)
        self.plt.setp(bp['caps'], color=color)
        self.plt.setp(bp['medians'], color=color)
        self.plt.setp(bp['fliers'], markeredgecolor=color)

    def do(self):
        self.plt.figure()

        offset = 1
        scale = 2 * 2
        n = (len(self.list_al_name)-1.0) / 2. * 0.7
        interval = np.arange(-n, n + 0.7, 0.7)
        width = 0.5
        dict_bp = {}
        
        for idx, al_name in enumerate(self.list_al_name):
            dict_bp[al_name] = self.plt.boxplot(self.dict_error[al_name], positions=np.array(range(len(self.ticks))) * scale + interval[idx] + offset, sym='+', widths=width)
            self._set_box_color_(dict_bp[al_name], self.color_set[idx])
            self.plt.plot([], c=self.color_set[idx], label=al_name)

        self.plt.legend()
        self.plt.xlabel("The number of the anchor sensors", fontsize=15)
        self.plt.ylabel("Error [m]", fontsize=15)
        self.plt.xticks(range(offset, scale*len(ticks), scale), self.ticks)
        self.plt.xlim(-scale/2, scale*len(self.ticks))
        self.plt.ylim(0, 0.3)
        self.plt.tight_layout()
        self.plt.savefig('boxcompare.png')
        self.plt.show()
        
class DataPlotModule:
    def __init__(self):
        self.plt = plt
        self.legend_title = '# of Sensors'
        self.y_axis_name = 'RMSE(cm)'
        self.x_axis_name = 'Sequence length'
        self.list_sequense_length = ['2', '3', '5', '8', '12']
        self.list_beacon_num = ['3', '5', '8']
        
        self.list_plt = []
        
    def set_data(self, np_data):
        '''
        :param np_data: raw = seq_len, column = beacon_num
        :return: 
        '''
        self.np_data = np_data

    def plot_data(self, np_data):
        self.set_data(np_data)
        self._make_plot_()
        self.plt.show()

    def _make_plot_(self):
        self.plt.figure()
        for idx in range(len(self.list_beacon_num)):
            self.list_plt.extend(self.plt.plot(self.list_sequense_length, self.np_data[idx], label=self.list_beacon_num[idx]))
        self._set_plot_option_()

    def _set_plot_option_(self):
        self.plt.legend(handles=self.list_plt, title=self.legend_title, loc=1)
        self.plt.xlabel(self.x_axis_name, fontsize=15)
        self.plt.ylabel(self.y_axis_name, fontsize=15)
        
        

if __name__ == '__main__':
    bpm = BoxPlotModule()
    bpm.do_test('RO_test/02-38.csv',['../result_data/Particle_filter','../result_data/fc','../result_data/stacked_bi', '../result_data/RONet'])
    # dpm = DataPlotModule()
    # dpm.plot_data([[3, 4, 1, 5, 9],[2, 3, 1, 6, 10],[0, 2, 5, 4, 1]])


