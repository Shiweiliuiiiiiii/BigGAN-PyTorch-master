import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


dense_acc_IP_0301 = []
dense_acc_FID_0301 = []
file = open('../results/Biggan_c10_03_01_DG')
for line in file:
    if 'PYTORCH UNOFFICIAL Inception Score' in line:
        print(line.split())
        dense_acc_IP_0301.append(float(line.split()[7]))
        dense_acc_FID_0301.append(float(line.split()[-1]))


dense_acc_IP_0305 = []
dense_acc_FID_0305 = []
file = open('../results/Biggan_c10_03_05')
for line in file:
    if 'PYTORCH UNOFFICIAL Inception Score' in line:
        print(line.split())
        dense_acc_IP_0305.append(float(line.split()[7]))
        dense_acc_FID_0305.append(float(line.split()[-1]))


dense_acc_IP_0805 = []
dense_acc_FID_0805 = []
file = open('../results/Biggan_c10_08_05')
for line in file:
    if 'PYTORCH UNOFFICIAL Inception Score' in line:
        print(line.split())
        dense_acc_IP_0805.append(float(line.split()[7]))
        dense_acc_FID_0805.append(float(line.split()[-1]))

dense_acc_IP_0105 = []
dense_acc_FID_0105 = []
file = open('../results/Biggan_c10_01_05')
for line in file:
    if 'PYTORCH UNOFFICIAL Inception Score' in line:
        print(line.split())
        dense_acc_IP_0105.append(float(line.split()[7]))
        dense_acc_FID_0105.append(float(line.split()[-1]))

fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
ax1.plot(np.arange(25), dense_acc_IP_0105, '-', label='0105', color='orange', linewidth = 2)
ax1.plot(np.arange(25), dense_acc_IP_0305, '-', label='0305', color='green', linewidth = 2)
ax1.plot(np.arange(25), dense_acc_IP_0805, '-', label='0805', color='blue', linewidth = 2)
ax1.plot(np.arange(25), dense_acc_IP_0301, '-', label='0301', color='black', linewidth = 2)

# ax1.plot(np.arange(25), dense_acc_FID_0105, '-', label='0105', color='orange', linewidth = 2)
# ax1.plot(np.arange(25), dense_acc_FID_0305, '-', label='0305', color='green', linewidth = 2)
# ax1.plot(np.arange(25), dense_acc_FID_0805, '-', label='0805', color='blue', linewidth = 2)
# ax1.plot(np.arange(25), dense_acc_FID_0301, '-', label='0301', color='black', linewidth = 2)
plt.legend(fontsize=20)
# plt.xticks(np.arange(25), ['0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
plt.xlabel('Density',fontsize=20)
plt.ylabel('Error',fontsize=20)
plt.show()