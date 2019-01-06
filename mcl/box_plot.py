import matplotlib.pyplot as plt
import numpy as np

def calculate_mean_RMSE(seq_result):
    mean = sum(seq_result)/3
    return mean

def set_y(seq_2_result, seq_3_result, seq_5_result, seq_7_result, seq_10_result):
    mean_2 = calculate_mean_RMSE(seq_2_result)
    mean_3 = calculate_mean_RMSE(seq_3_result)
    mean_5 = calculate_mean_RMSE(seq_5_result)
    mean_7 = calculate_mean_RMSE(seq_7_result)
    mean_10 = calculate_mean_RMSE(seq_10_result)

    return np.array([mean_2, mean_3, mean_5, mean_7, mean_10])

x = np.array([2,3,5,7,10])


''''''
seq_2_result = [2, 3, 4]
seq_3_result = [3, 4, 6]
seq_5_result = [4, 5, 8]
seq_7_result = [5, 6, 10]
seq_10_result = [6, 7, 12]
''''''



y = set_y(seq_2_result, seq_3_result, seq_5_result, seq_7_result, seq_10_result)

lower_error = [y[0] - seq_2_result[0], y[1] - seq_3_result[0], y[2] - seq_5_result[0],  y[3] - seq_7_result[0], y[4] - seq_10_result[0]]
upper_error = [seq_2_result[2] - y[0], seq_3_result[2] - y[1], seq_5_result[2] - y[2], seq_7_result[2] - y[3], seq_10_result[2] - y[4]]
# upper_error = [1]*5
z = [lower_error, upper_error]
saved_file_name = "seq_result.png"
plot_title = "RMSE According to Sequence Length"
plt.title(plot_title)
# plt.figure(figsize=(7,4.326))


plt.errorbar(x, y, yerr= z, fmt = '-^', capsize=4, color=(7/255,109/255,173/255))

plt.grid(True)
# plt.xlim(0,1000)
# plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
# plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
# plt.ylim(0.0,30)
plt.xlabel("Seqeuence Length" ,fontsize=14)
plt.ylabel("RMSE [cm]", fontsize= 14)
fig = plt.gcf()
plt.show()
fig.savefig(saved_file_name)
print ("Done")

