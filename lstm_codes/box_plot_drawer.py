import matplotlib.pyplot as plt
import numpy as np


data_pozyx = [[1,2,5], [5,7,2,2,5], [7,2,5]]
data_RO_SLAM = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
data_MLP = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
data_BI_LSTM = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
    data_OURS = [range(1,10), [1,2,5,3,2], [2,3,5,1]]

ticks = ['3', '5', '8']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

offset = 1
scale = 2
interval = 0.7
width = 0.5

bp1 = plt.boxplot(data_pozyx, positions=np.array(range(len(data_pozyx)))*2.0*scale-2*interval + offset, sym='', widths=width)
bp2 = plt.boxplot(data_RO_SLAM, positions=np.array(range(len(data_RO_SLAM)))*2.0*scale-interval + offset, sym='', widths=width)
bp3 = plt.boxplot(data_MLP, positions=np.array(range(len(data_MLP)))*2.0*scale + offset, sym='', widths=width)
bp4 = plt.boxplot(data_BI_LSTM, positions=np.array(range(len(data_BI_LSTM)))*2.0*scale+interval + offset, sym='', widths=width)
bp5 = plt.boxplot(data_OURS, positions=np.array(range(len(data_OURS)))*2.0*scale+2*interval + offset, sym='', widths=width)
# bpk = plt.boxplot(data_c, positions=np.array(range(len(data_c)))*2.0+0.4, sym='', widths=0.6)

color_set = ['#34B291', '#FFC000', '#EE3030', '#30EE30', '#3030EE']

set_box_color(bp1, color_set[0]) # colors are from http://colorbrewer2.org/
set_box_color(bp2, color_set[1])
set_box_color(bp3, color_set[2])
set_box_color(bp4, color_set[3])
set_box_color(bp5, color_set[4])

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c=color_set[0], label='Pozyx')
plt.plot([], c=color_set[1], label='RO SLAM')
plt.plot([], c=color_set[2], label='MLP')
plt.plot([], c=color_set[3], label='Bi-LSTM')
plt.plot([], c=color_set[4], label='Ours')
# plt.plot([], c='#007BB0', label='Test')
plt.legend()
plt.xlabel("The number of the anchor sensors", fontsize=15)
plt.ylabel("Error [cm]", fontsize=15)
plt.xticks(range(offset, scale*len(ticks)*2, 2*scale), ticks)
plt.xlim(-2, scale*len(ticks)*2)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig('boxcompare.png')