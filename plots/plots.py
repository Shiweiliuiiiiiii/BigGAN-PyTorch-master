import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

fig=figure(num=None, figsize=(16, 4), dpi=120, facecolor='w', edgecolor='k')
fontsize = 15

fontsize = 15
# seed0 seed1 seed2
# fix various Density: 0.5 ratio, Density = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# BigGAN_fix_all_sparsity = [18.5010, 13.829150, 10.186845, 9.009996, 8.883314, 8.447915, 8.822600, 8.552754, 8.123241, 9.254222]

#Fix density=0.3, various ratio: 0.05, 0.1,  0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90
#Fix density=0.3, density of G : 0.03, 0.06, 0.12, 0.20, 0.31

BigGAN_fix_03_imbalamced = [27.746760, 16.44, 13.309706, 11.449772, 9.009996, 9.755407 ,11.165939, 11.303984, 13.898175,16.32]
#DST density=0.3, various ratio: 0.05, 0.1, 0.20, 0.30, 0.50
BigGAN_DST_03_imbalamced = [17.929470, 10.940174, 10.453464, 9.190630, 8.848089] #  gradient
# BigGAN_DST_03_imbalamced = [ 16.320349, 11.034435, 10.253493, 8.613544，9.479591] #  random

# FIX, dst: DG, D, G
BigGAN_DST_0305 = [9.009996, 8.848089, 8.202398, 9.458214] #G-S=0.3
BigGAN_DST_0301 = [16.44, 10.940174, 18.716372, 11.732190] #G-S=0.06; D-S=0.51    when G is extremely sparse, we can use DST to bost performance

#BigGAN_DST_03005, 03010 ,03016, 03020, 03030, 03050
BigGAN_DST_03 = [17.929470, 10.940174, 10.349429, 10.453464, 8.848089]

# BigGAN_fix_all_sparsity = BigGAN_fix_all_sparsity[::-1]

# finetune update frequency # 0.05   fre=500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 50000  # seed0
BigGAN_DST_random = [14.229831, 15.498680, 13.991855, 13.718603, 12.331170, 17.809324, 12.099610, 15.488815, 15.457775, 18.095]
BigGAN_DST_gradient = [                                17.545861, 18.223253, 18.545319, 15.634578, 14.876048, 15.293339, 17.069357]


# pure pruning sparsity=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05] seed0
BigGAN_prue_pruning = [8.365654, 8.423706, 8.581143, 9.408, 10.721, 16.06, 21.18, 39.674, 62.0477, 103.44, 155.756365]
# prune and finetune balancely seed1
BigGAN_FT_s1 = [8.453215, 8.514303, 8.541244, 8.497485, 8.501285,  8.408194, 8.398978,  8.339861, 8.605778, 9.624805, 10.492408]

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# -----------------------biggan--------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# dense trianing
Dense = np.array([9.02, 8.364970, 8.998160])
Dense_mean = Dense.mean()
Dense_std = Dense.std()

# balanced training
# seed1 balanced fixed density is [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
Biggan_balanced_s2 = [8.732128, 8.742346, 9.252907, 8.848907, 7.974577, 9.534973,  9.117283, 10.807817, 15.266711, 18.267072]
Biggan_balanced_s1 = [8.732143, 8.742346, 8.168308, 9.452966, 8.847022, 9.182337,  9.302774, 12.275600, 13.685525, 18.409937]
Biggan_balanced_s0 = [18.5010, 13.829150, 10.186845, 9.009996, 8.883314, 8.447915, 8.822600, 8.552754, 8.123241, 9.254222]  # liushiwei1 seed 0
Biggan_balanced_s0 = Biggan_balanced_s0[::-1]
Biggan_balanced = np.stack((Biggan_balanced_s0, Biggan_balanced_s1, Biggan_balanced_s2))
Biggan_balanced_mean = Biggan_balanced.mean(axis=0)
Biggan_balanced_std = Biggan_balanced.std(axis=0)
print('Biggan_balanced_mean:', Biggan_balanced_mean)

Biggan_DST_gradient = [15.785479, 12.321365, 13.107435, 13.797741]
Biggan_DST_random = [14.385917, 12.469672, 13.563525,  13.607501]
# [0.02 0.05 0.1 0.2 0.3 0.4] fixed
Biggan_unbalanced_s1 = [32.233507, 17.435703, 12.755465, 12.360272, 9.652078, 9.840711]
Biggan_unbalanced_s2 = [31.553134, 18.349253, 13.334879, 12.130157, 9.718945, 12.695755]

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# -----------------------sngan------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# dense trianing
Dense_sngan = np.array([13.3345])
Dense_sngan_mean = Dense_sngan.mean()
Dense_sngan_std = Dense_sngan.std()

# balanced training
# seed1 balanced fixed density is [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
# sngan_balanced_s2 = [8.732128, 8.742346, 9.252907, 8.848907, 7.974577, 9.534973,  9.117283, 10.807817, 15.266711, 18.267072]
sngan_balanced_s1 = [13.6897, 13.7728, 14.1200, 13.7460 ,14.7500, 15.7746, 17.3062, 21.1183, 30.5370, 36.1987]
# sngan_balanced_s0 = [18.5010, 13.829150, 10.186845, 9.009996, 8.883314, 8.447915, 8.822600, 8.552754, 8.123241, 9.254222]  # liushiwei1 seed 0

# sngan_balanced = np.stack((sngan_balanced_s0, sngan_balanced_s1, sngan_balanced_s2))
# sngan_balanced_mean = sngan_balanced.mean(axis=0)
# sngan_balanced_std = sngan_balanced.std(axis=0)
# print('Biggan_balanced_mean:', sngan)


markersize = '5'
alpha = 0.3
fontsize=15
ticksize=10
X_axis = np.arange(len(Biggan_balanced_mean))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(X_axis, [Dense_mean] * 10, '--', color='brown', linewidth = 2)
ax1.plot(X_axis, Biggan_balanced_mean, '-', label='CIFAR-10', color='brown', linewidth = 2)
ax1.fill_between(X_axis,  Biggan_balanced_mean+Biggan_balanced_std, Biggan_balanced_mean-Biggan_balanced_std,color='brown',alpha=alpha,linewidth=0)
plt.xticks(np.arange(10), [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax1.tick_params(axis='both', which='minor', labelsize=fontsize)
ax1.set_ylabel('FID',fontsize=fontsize)
ax1.set_xlabel('Remaining weights (%)',fontsize=fontsize)
ax1.grid(True, linestyle='-', linewidth=0.5, )
ax1.set_title('Balanced BigGAN',fontsize=fontsize)
ax1.legend(fontsize=fontsize)
# remove top and right spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2 = fig.add_subplot(1,3,2)
ax2.plot(X_axis, [Dense_sngan] * 10, '--', color='brown', linewidth = 2)
ax2.plot(X_axis, sngan_balanced_s1, '-', label='CIFAR-10', color='brown', linewidth = 2)
# ax2.fill_between(X_axis,  Biggan_balanced_mean+Biggan_balanced_std, Biggan_balanced_mean-Biggan_balanced_std,color='brown',alpha=alpha,linewidth=0)
plt.xticks(np.arange(10), [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax2.tick_params(axis='both', which='minor', labelsize=fontsize)
ax2.set_ylabel('FID',fontsize=fontsize)
ax2.set_xlabel('Remaining weights (%)',fontsize=fontsize)
ax2.grid(True, linestyle='-', linewidth=0.5, )
ax2.set_title('Balanced SNGAN',fontsize=fontsize)
ax2.legend(fontsize=fontsize)
# remove top and right spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax3 = fig.add_subplot(1,3,3)
ax3.plot(X_axis, [Dense_mean] * 10, '--', color='brown', linewidth = 2)
ax3.plot(X_axis, Biggan_balanced_mean, '-', label='horse2zebra', color='brown', linewidth = 2)
ax3.fill_between(X_axis,  Biggan_balanced_mean+Biggan_balanced_std, Biggan_balanced_mean-Biggan_balanced_std,color='brown',alpha=alpha,linewidth=0)
plt.xticks(np.arange(10), [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax3.tick_params(axis='both', which='minor', labelsize=fontsize)
ax3.set_ylabel('FID',fontsize=fontsize)
ax3.set_xlabel('Remaining weights (%)',fontsize=fontsize)
ax3.grid(True, linestyle='-', linewidth=0.5, )
ax3.legend(fontsize=fontsize)
ax3.set_title('Balanced CycleGAN',fontsize=fontsize)
# remove top and right spines
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)



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

plt.show()