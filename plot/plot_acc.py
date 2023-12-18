import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils_rl as utils
from scipy.linalg import orthogonal_procrustes
from numpy.linalg import norm
mpl.use('Qt5Agg')
font = {'size': 12}
mpl.rc('font', **font)
plt.ion()

crlb = np.load('./../output/crlb_acc3.npz')
crlb_K = crlb['K']
crlb_main_y0 = crlb['rmse_main_y0']
crlb_main_y1 = crlb['rmse_main_y1']
crlb_main_y1_ctr = crlb['rmse_main_y1_ctr']
crlb_main_y2 = crlb['rmse_main_y2']
crlb_main_b0 = crlb['rmse_main_b0']
crlb_main_b1 = crlb['rmse_main_b1']
crlb_main_b2 = crlb['rmse_main_b2']
crlb_main_b3 = crlb['rmse_main_b3']
crlb_main_b4 = crlb['rmse_main_b4']

mc_runs = np.load('./../output/mc_2step_acc.npz')
rmse_K = mc_runs['K']
rmse_y0_hat = mc_runs['rmse_y0_hat']
rmse_y1_hat = mc_runs['rmse_y1_hat']
rmse_y2_hat = mc_runs['rmse_y2_hat']
rmse_b0_hat = mc_runs['rmse_b0_hat']
rmse_b1_hat = mc_runs['rmse_b1_hat']
rmse_b2_hat = mc_runs['rmse_b2_hat']
rmse_b3_hat = mc_runs['rmse_b3_hat']
rmse_b4_hat = mc_runs['rmse_b4_hat']

rmse_y0_acc = mc_runs['rmse_y0_acc']
rmse_y1_acc = mc_runs['rmse_y1_acc']
rmse_y2_acc = mc_runs['rmse_y2_acc']
rmse_b0_acc = mc_runs['rmse_b0_acc']
rmse_b1_acc = mc_runs['rmse_b1_acc']
rmse_b2_acc = mc_runs['rmse_b2_acc']
rmse_b3_acc = mc_runs['rmse_b3_acc']

'''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DISCLAIMER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    I made a mistake saving err_y1_acc as it is the same as err_y1_hat.
    I'll redo the calculations here rather than running the entire algorithm
'''
Y0 = np.array([[-244.0, -588.0], [385.0, -456.0], [81.0, -992.0], [-19.0, -730.0], [-792.0, 879.0], [-554.0, 970.0],
                   [-965.0, 155.0], [-985.0, 318.0], [-49.0, -858.0], [-503.0, 419.0]]).transpose()
Y1 = np.array(
        [[-5.0, -8.0], [-8.0, -5.0], [-6.0, -7.0], [6.0, -9.0], [-1.0, -3.0], [2.0, -2.0], [1.0, -2.0], [-5.0, -10.0],
         [9.0, 2.0], [-5.0, -1.0]]).transpose()
Y2 = np.array(
        [[-0.17, 0.42], [-0.42, 0.17], [0.22, 0.98], [-0.07, 0.73], [0.21, 0.48], [-0.15, 0.08], [0.55, -0.43],
         [-0.72, -0.14], [-0.49, 0.56], [-0.34, 0.91]]).transpose()

Y1_bar_acc = mc_runs['Y1_bar_acc']
nDim, N = Y0.shape
one = np.ones((N, 1))
C = np.eye(N) - (one @ one.T) / N
Y1_bar = Y1 @ C

def procrustes_error(Z, Z_bar):
    H, scale = orthogonal_procrustes(Z.T, Z_bar.T)
    Z_proc = Z.T @ H
    err_z = utils.vectorize(Z_bar - Z_proc.T)

    return np.squeeze(err_z), H

_, _, N_EXP, K = Y1_bar_acc.shape
err_y1_acc = np.zeros((nDim * N, N_EXP, K))
for kk in range(K):
    for nn in range(N_EXP):
        err_y1_acc[:, nn, kk], H_Y1 = procrustes_error(Y1_bar_acc[:, :, nn, kk], Y1_bar)
    rmse_y1_acc[kk] = np.sqrt(np.sum(np.square(norm(err_y1_acc[:, :, kk], axis=0))) / N_EXP) / (nDim * N)

lw = 3.0
fig, axs = plt.subplots(3, 1, gridspec_kw={'hspace': 0.3})
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')

axs[0].plot(rmse_K, rmse_y0_hat, 'bo-', linewidth=lw, label="w/o accelerometer")
axs[0].plot(rmse_K, rmse_y0_acc, 'ro-', linewidth=lw, label="w/ accelerometer")
axs[0].set_ylabel('RMSE [$m$]')
axs[0].set_title('Relative Position $\mathbf{Y}_0$', fontsize=12)
# plt.setp(axs[0], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs[0].xaxis.set_ticklabels([])
plt.setp(axs[0], yticks=[1e-3, 1e-4])
# axs[0].set_xlabel('No. of timestamps [K]')
# axs[0].set_xlabel('No. of timestamps [K]')
axs[0].legend()
axs[0].grid()

axs[1].plot(rmse_K, rmse_y1_hat, 'bo-', linewidth=lw, label="w/o accelerometer")
axs[1].plot(rmse_K, rmse_y1_acc, 'ro-', linewidth=lw, label="w/ accelerometer")
axs[1].set_ylabel('RMSE [$m/s$]')
axs[1].set_title('Relative Velocity $\mathbf{Y}_1$', fontsize=12)
# plt.setp(axs[1], yticks=[1e-3, 1e-4])
# plt.setp(axs[1], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs[1].xaxis.set_ticklabels([])
# axs[1].legend()
axs[1].grid()

axs[2].plot(rmse_K, rmse_y2_hat, 'bo-', linewidth=lw, label="w/o accelerometer")
axs[2].plot(rmse_K, rmse_y2_acc, 'ro-', linewidth=lw, label="w/ accelerometer")
axs[2].set_ylabel('RMSE [$m/s^2$]')
axs[2].set_title('Relative Acceleration $\mathbf{Y}_2$', fontsize=12)
axs[2].set_xlabel('No. of timestamps [K]')
# plt.setp(axs[2], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# axs[2].legend()
axs[2].grid()

fig3, axs3 = plt.subplots(3, 1)
axs3[0].set_yscale('log')
axs3[1].set_yscale('log')
axs3[2].set_yscale('log')

axs3[0].plot(rmse_K, rmse_y0_hat, 'bo-', linewidth=lw, label="proposed")
axs3[0].plot(rmse_K, crlb_main_y0[:, 2], 'rx--', linewidth=lw, label="CRLB")
axs3[0].set_ylabel('RMSE [$m$]')
axs3[0].set_title('Relative Position $\mathbf{Y}_0$', fontsize=12)
plt.setp(axs3[0], yticks=[1e-3, 1e-4])
plt.setp(axs3[0], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs3[0].xaxis.set_ticklabels([])
# axs[0]3.set_xlabel('No. of timestamps [K]')
axs3[0].legend()
axs3[0].grid()

axs3[1].plot(rmse_K, rmse_y1_hat, 'bo-', linewidth=lw, label="proposed")
axs3[1].plot(rmse_K, crlb_main_y1[:, 2], 'rx--', linewidth=lw, label="CRLB")
axs3[1].plot(rmse_K, crlb_main_y1_ctr[:, 2], 'g+--', linewidth=lw, label="Constrained CRLB")
axs3[1].set_ylabel('RMSE [$m/s$]')
axs3[1].set_title('Relative Velocity $\mathbf{Y}_1$', fontsize=12)
plt.setp(axs3[1], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs3[1].xaxis.set_ticklabels([])
axs3[1].legend()
axs3[1].grid()

axs3[2].plot(rmse_K, rmse_y2_hat, 'bo-', linewidth=lw, label="proposed")
axs3[2].plot(rmse_K, crlb_main_y2[:, 2], 'rx--', linewidth=lw, label="CRLB")
axs3[2].set_ylabel('RMSE [$m/s^2$]')
axs3[2].set_title('Relative Acceleration $\mathbf{Y}_2$', fontsize=12)
axs3[2].set_xlabel('No. of timestamps [K]')
plt.setp(axs3[2], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs3[2].legend()
axs3[2].grid()

plt.show()

print('finished!')

