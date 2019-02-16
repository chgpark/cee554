import numpy as np

def get_std(m1, m2, m3, m4, idx):
    sensor_1m = m1[:, idx]
    sensor_2m = m2[:, idx]
    sensor_3m = m3[:, idx]
    sensor_4m = m4[:, idx]

    std_1m = np.std(sensor_1m)
    std_2m = np.std(sensor_2m)
    std_3m = np.std(sensor_3m)
    std_4m = np.std(sensor_4m)

    print(std_1m, std_2m, std_3m, std_4m)

if __name__ == "__main__":
    file_name = '5678'
    csv_1 = file_name + '_1m.csv'
    csv_2 = file_name + '_2m.csv'
    csv_3 = file_name + '_3m.csv'
    csv_4 = file_name + '_4m.csv'

    m1 = np.loadtxt(csv_1, delimiter=',')
    m2 = np.loadtxt(csv_2, delimiter=',')
    m3 = np.loadtxt(csv_3, delimiter=',')
    m4 = np.loadtxt(csv_4, delimiter=',')
    """
    Select target
    0, 2, 4, 6
    """
    get_std(m1, m2, m3, m4, 6)

