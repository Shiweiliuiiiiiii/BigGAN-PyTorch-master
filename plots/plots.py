import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

fontsize = 15


# fix various Density: 0.5 ratio, Density = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BigGAN_fix_all_sparsity = [18.5010, 13.829150, 10.186845, 9.009996, 8.883314, 8.447915, 8.822600, 8.552754,8.123241,9.254222]

#Fix density=0.3, various ratio: 0.05, 0.1, 0.20, 0.30, 0.50
BigGAN_fix_03_imbalamced = [27.746760, 16.44, 13.309706, 11.449772, 9.009996]
#DST density=0.3, various ratio: 0.05, 0.1, 0.20, 0.30, 0.50
BigGAN_DST_03_imbalamced = [17.929470, 10.940174, 10.453464, 9.190630, 8.848089] #  gradient
BigGAN_DST_03_imbalamced = [16.282417, ]

# FIX, dst: DG, D, G
BigGAN_DST_0305 = [9.009996, 8.848089, 8.202398, 9.458214] #G-S=0.3
BigGAN_DST_0301 = [16.44, 10.940174, 18.716372, 11.732190] #G-S=0.06; D-S=0.51    when G is extremely sparse, we can use DST to bost performance

#BigGAN_DST_03005, 03010 ,03016, 03020, 03030, 03050
BigGAN_DST_03 = [17.929470, 10.940174, 10.349429, 10.453464, 8.848089]

BigGAN_fix_all_sparsity = BigGAN_fix_all_sparsity[::-1]
# BigGAN_line = np.stack((BigGAN_01[-1],BigGAN_03[-1],BigGAN_05[-1],BigGAN_07[-1],BigGAN_09[-1]))

# BigGAN_line_mean = BigGAN_line.mean(axis=1)
# BigGAN_line_std = BigGAN_line.std(axis=1)

# neme = ['RN20','RN32','RN44','RN56']
fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
ax1.plot(np.arange(5), BigGAN_fix_03_imbalamced, '-', label='Static Sparse BigGAN with 30% parameters', color='orange', linewidth = 2)
ax1.plot(np.arange(5), BigGAN_DST_03_imbalamced, '-', label='Dynamic Sparse BigGAN with 30% parameters', color='blue', linewidth = 2)
# ax1.fill_between(np.arange(5), BigGAN_line_mean - BigGAN_line_std, BigGAN_line_mean + BigGAN_line_std, alpha=0.2, color='orange')
# ax1.plot(0.27, acc_rn20_dense_mean, 'o', color='orange', label='RN20_dense')
ax1.plot(np.arange(5), [9.02]*5, '-', label='dense BigGAN', color='black')
# ax1.plot(np.arange(10), BigGAN_IP, '--', label='Sparse BigGAN IP', color='orange')
# ax1.fill_between(np.arange(9),  [acc_rn20_dense_mean-acc_rn20_dense_std]*9, [acc_rn20_dense_mean+acc_rn20_dense_std]*9, alpha=0.2, color='black')
plt.xticks(np.arange(5), ['0.05 (G_density=0.03)','0.1 (G_density=0.06)','0.2 (G_density=0.12)','0.3 (G_density=0.20)','0.5 (G_density=0.31)'],fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Parameter count ratio between G and D',fontsize=fontsize)
plt.title('Imbalanced Training GAN')
plt.ylabel('FID',fontsize=fontsize)
# plt.ylim(87.5, 94)
# plt.savefig('Effect of parameter ratio on the sparse BigGAN performance')
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