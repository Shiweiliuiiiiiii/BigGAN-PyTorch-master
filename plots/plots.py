import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

fontsize = 15
# ratio 0.1 0.3 0.5 0.7 0.9
BigGAN_01 = np.array([10.244079, 9.377331, 8.547908, 8.514451, 8.335155]) # 65000 iteration
BigGAN_03 = np.array([13.6162, 12.0273, 10.9210, 10.182, 7.8727]) # 85000
BigGAN_05 = np.array([10.6258, 8.3983, 8.3352, 8.234478, 8.330472]) # 85000
BigGAN_07 = np.array([8.836767, 8.6992, 8.6248, 8.5837, 8.4033]) # 80000
BigGAN_09 = np.array([8.0230, 7.9954, 7.9587, 7.9245, 7.7857 ])  # 75000 iteration

BigGAN_line = np.stack((BigGAN_01,BigGAN_03,BigGAN_05,BigGAN_07,BigGAN_09))

BigGAN_line_mean = BigGAN_line.mean(axis=1)
BigGAN_line_std = BigGAN_line.std(axis=1)



# neme = ['RN20','RN32','RN44','RN56']
fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
ax1.plot(np.arange(5), BigGAN_line_mean, '-', label='70% sparsity BigGAN', color='orange')
ax1.fill_between(np.arange(5), BigGAN_line_mean - BigGAN_line_std, BigGAN_line_mean + BigGAN_line_std, alpha=0.2, color='orange')
# ax1.plot(0.27, acc_rn20_dense_mean, 'o', color='orange', label='RN20_dense')
ax1.plot(np.arange(5), [9.02]*5, '-', label='dense BigGAN', color='black')
# ax1.fill_between(np.arange(9),  [acc_rn20_dense_mean-acc_rn20_dense_std]*9, [acc_rn20_dense_mean+acc_rn20_dense_std]*9, alpha=0.2, color='black')
plt.xticks(np.arange(5), ['ratio=0.1 G_density = 0.06', 'ratio=0.3 G_density = 0.17', 'ratio=0.5 G_density = 0.32','ratio=0.7 G_density = 0.45', 'ratio=0.9 S_G = 0.57',],fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('ratio = #G/#GD, G_density',fontsize=fontsize)
plt.ylabel('FID',fontsize=fontsize)
# plt.ylim(87.5, 94)
plt.savefig('Effect of parameter ratio on the sparse BigGAN performance')

# ax1.plot(np.arange(9), acc_rn32_rp_mean, '-', label='RN32_Random_Pruning', color='orange')
# ax1.fill_between(np.arange(9), acc_rn32_rp_mean - acc_rn32_rp_std, acc_rn32_rp_mean + acc_rn32_rp_std, alpha=0.2, color='orange')
# # ax1.plot(0.46, acc_rn32_dense_mean, 'o', color='green', label='RN32_dense')
# ax1.plot(np.arange(9), [acc_rn32_dense_mean]*9, '-', label='RN32', color='black')
# ax1.fill_between(np.arange(9),  [acc_rn32_dense_mean-acc_rn32_dense_std]*9, [acc_rn32_dense_mean+acc_rn32_dense_std]*9, alpha=0.2, color='black')
# plt.xticks(np.arange(9), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlabel('Density',fontsize=fontsize)
# plt.ylabel('Test Acc',fontsize=fontsize)
# plt.ylim(87.5, 94)
# plt.savefig('RN32')

# ax1.plot(np.arange(9), acc_rn44_rp_mean, '-', label='RN44_Random_Pruning', color='orange')
# ax1.fill_between(np.arange(9), acc_rn44_rp_mean - acc_rn44_rp_std, acc_rn44_rp_mean + acc_rn44_rp_std, alpha=0.2, color='orange')
# # ax1.plot(0.66, acc_rn44_dense_mean, 'o', color='blue', label='RN44_dense')
# ax1.plot(np.arange(9), [acc_rn44_dense_mean]*9, '-', label='RN44', color='black')
# ax1.fill_between(np.arange(9),  [acc_rn44_dense_mean-acc_rn44_dense_std]*9, [acc_rn44_dense_mean+acc_rn44_dense_std]*9, alpha=0.2, color='black')
# plt.xticks(np.arange(9), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlabel('Density',fontsize=fontsize)
# plt.ylabel('Test Acc',fontsize=fontsize)
# plt.ylim(87.5, 94)
# plt.savefig('RN44')

# ax1.plot(np.arange(9), acc_rn56_rp_mean, '-', label='RN56_Random_Pruning', color='orange')
# ax1.fill_between(np.arange(9), acc_rn56_rp_mean - acc_rn56_rp_std, acc_rn56_rp_mean + acc_rn56_rp_std, alpha=0.2, color='orange')
# # ax1.plot(0.85, acc_rn56_dense_mean, 'o', color='blue')
# ax1.plot(np.arange(9), [acc_rn56_dense_mean]*9, '-', label='RN56', color='black')
# ax1.fill_between(np.arange(9),  [acc_rn56_dense_mean-acc_rn56_dense_std]*9, [acc_rn56_dense_mean+acc_rn56_dense_std]*9, alpha=0.2, color='black')
# plt.xticks(np.arange(9), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlabel('Density',fontsize=fontsize)
# plt.ylabel('Test Acc',fontsize=fontsize)
# plt.ylim(87.5, 94)
# plt.savefig('RN56')

# ax1.plot(np.arange(9), acc_rn110_rp_mean, '-', label='RN110', color='orange')
# ax1.fill_between(np.arange(9), acc_rn110_rp_mean - acc_rn110_rp_std, acc_rn110_rp_mean + acc_rn110_rp_std, alpha=0.2, color='orange')
# ax1.plot(np.arange(9), [acc_rn110_dense_mean]*9, '-', label='RN110', color='black')
# ax1.fill_between(np.arange(9),  [acc_rn110_dense_mean-acc_rn110_dense_std]*9, [acc_rn110_dense_mean+acc_rn110_dense_std]*9, alpha=0.2, color='black')
# plt.xticks(np.arange(9), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlabel('Density',fontsize=fontsize)
# plt.ylabel('Test Acc',fontsize=fontsize)
# plt.ylim(87.5, 94)
# plt.savefig('RN110')

# plt.xlabel('Number of parameters (M)')
# plt.ylabel('Test Acc')
plt.legend(fontsize=fontsize)
plt.show()