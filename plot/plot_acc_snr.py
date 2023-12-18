import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
from scipy.linalg import orthogonal_procrustes
from numpy.linalg import norm
mpl.use('Qt5Agg')
plt.ion()
font = {'size'   : 12}
mpl.rc('font', **font)

mc_runs = np.load('./../output/mc_2step_acc_snr.npz')
SNR = -mc_runs['snr']
rmse_y0_hat = mc_runs['rmse_y0_hat']
rmse_y1_hat = mc_runs['rmse_y1_hat']
rmse_y2_hat = mc_runs['rmse_y2_hat']
rmse_b0_hat = mc_runs['rmse_b0_hat']
rmse_b1_hat = mc_runs['rmse_b1_hat']
rmse_b2_hat = mc_runs['rmse_b2_hat']
rmse_b3_hat = mc_runs['rmse_b3_hat']
rmse_b4_hat = mc_runs['rmse_b4_hat']
err_y1_hat = mc_runs['err_y1_hat']

rmse_y0_acc = mc_runs['rmse_y0_acc']
rmse_y1_acc = mc_runs['rmse_y1_acc']
rmse_y2_acc = mc_runs['rmse_y2_acc']
rmse_b0_acc = mc_runs['rmse_b0_acc']
rmse_b1_acc = mc_runs['rmse_b1_acc']
rmse_b2_acc = mc_runs['rmse_b2_acc']
rmse_b3_acc = mc_runs['rmse_b3_acc']
err_y1_acc = mc_runs['err_y1_acc']

# recalculating err_ynd_hat properly
N, N_EXP, M = err_y1_hat.shape
err_ynd_hat = np.zeros((2, N_EXP, M))
err_ynd_acc = np.zeros((2, N_EXP, M))
err_yd_hat = np.zeros((N - 2, N_EXP, M))
err_yd_acc = np.zeros((N - 2, N_EXP, M))
rmse_ynd_hat = np.zeros(M)
rmse_ynd_acc = np.zeros(M)
rmse_yd_hat = np.zeros(M)
rmse_yd_acc = np.zeros(M)

for jj in range(M):
    for nn in range(N_EXP):
        # contains the off diagonal elements involved in ls
        err_ynd_hat[:, nn, jj] = np.squeeze(np.vstack((err_y1_hat[1, nn, jj], err_y1_hat[2, nn, jj])))
        err_ynd_acc[:, nn, jj] = np.squeeze(np.vstack((err_y1_acc[1, nn, jj], err_y1_acc[2, nn, jj])))

        # contains everything apart from the off diagonal elements involved in ls
        tmp = np.squeeze(np.vstack((err_y1_hat[0, nn, jj], err_y1_hat[3, nn, jj])))
        err_yd_hat[:, nn, jj] = np.concatenate((tmp, err_y1_hat[4:, nn, jj]))

        tmp = np.squeeze(np.vstack((err_y1_acc[0, nn, jj], err_y1_acc[3, nn, jj])))
        err_yd_acc[:, nn, jj] = np.concatenate((tmp, err_y1_acc[4:, nn, jj]))

    rmse_ynd_hat[jj] = np.sqrt(np.sum(np.square(norm(err_ynd_hat[:, :, jj], axis=0))) / N_EXP) / err_ynd_hat.shape[0]
    rmse_ynd_acc[jj] = np.sqrt(np.sum(np.square(norm(err_ynd_acc[:, :, jj], axis=0))) / N_EXP) / err_ynd_acc.shape[0]
    rmse_yd_hat[jj] = np.sqrt(np.sum(np.square(norm(err_yd_hat[:, :, jj], axis=0))) / N_EXP) / err_yd_hat.shape[0]
    rmse_yd_acc[jj] = np.sqrt(np.sum(np.square(norm(err_yd_acc[:, :, jj], axis=0))) / N_EXP) / err_yd_acc.shape[0]

# outlier removal - only extreme values
# recalculating err_ynd_hat properly
# err_y1_hat_wo = np.zeros((N, N_EXP, M))
threshold = 10
out_indices = [[] for _ in range(M)]

for kk in range(M):
    for ii in range(N):
        # contains the off diagonal elements involved in ls
        out_indices[kk].append(np.where(np.abs(err_y1_hat[ii, :, kk]) >= threshold))

idx_2_remove = [[] for _ in range(M)]
for ii in range(M):
    for jj in range(len(out_indices[ii])):
        # print([ii, jj])
        if len(out_indices[ii][jj][0]):
            for kk in range(len(out_indices[ii][jj][0])):
                if out_indices[ii][jj][0][kk] not in idx_2_remove[ii]:
                    idx_2_remove[ii].append(out_indices[ii][jj][0][kk])

err_y1_hat_no_outlier = []#[[] for _ in range(M)]
rmse_y1_hat_no_outlier = np.zeros(M)
for ii in range(M):
    if len(idx_2_remove[ii]):
        err_y1_hat_no_outlier.append(np.delete(err_y1_hat[:, :, ii], idx_2_remove[ii], 1))
    else:
        err_y1_hat_no_outlier.append(err_y1_hat[:, :, ii])
    tmp_N = err_y1_hat_no_outlier[ii].shape[1]
    print([ii, tmp_N])
    rmse_y1_hat_no_outlier[ii] = np.sqrt(np.sum(np.square(norm(err_y1_hat_no_outlier[ii], axis=0))) / tmp_N) / N

def procrustes_error(Z, Z_bar):
    H, scale = orthogonal_procrustes(Z.T, Z_bar.T)
    Z_proc = Z.T @ H
    err_z = utils.vectorize(Z_bar - Z_proc.T)

    return np.squeeze(err_z), H

lw = 3.0
fig, axs = plt.subplots(3, 1)
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')

axs[0].plot(SNR, rmse_y0_hat, 'bo-', linewidth=lw, label="w/o accelerometer")
axs[0].plot(SNR, rmse_y0_acc, 'ro-', linewidth=lw, label="w/ accelerometer")
axs[0].set_title('Relative Position $\mathbf{Y}_0$')
plt.setp(axs[0], xticks=[0, 5, 10, 15, 20, 25, 30])
axs[0].legend()
axs[0].xaxis.set_ticklabels([])
axs[0].set_ylabel('RMSE [$m$]')
plt.setp(axs[0], yticks=[1e-2, 1e-3, 1e-4, 1e-5])
axs[0].grid()

axs[1].plot(SNR, rmse_y1_hat, 'bo-', linewidth=lw, label="w/o accelerometer")
# axs[1].plot(SNR, rmse_y1_hat_no_outlier, 'bo-', linewidth=lw, label="w/o accelerometer")
axs[1].plot(SNR, rmse_y1_acc, 'ro-', linewidth=lw, label="w/ accelerometer")
axs[1].set_title('Relative Velocity $\mathbf{Y}_1$')
# axs[1].legend()
axs[1].xaxis.set_ticklabels([])
axs[1].set_ylabel('RMSE [$m/s$]')
plt.setp(axs[1], xticks=[0, 5, 10, 15, 20, 25, 30])
plt.setp(axs[1], yticks=[10, 1e-1, 1e-3])
axs[1].grid()

axs[2].plot(SNR, rmse_y2_hat, 'bo-', linewidth=lw, label="w/o accelerometer")
axs[2].plot(SNR, rmse_y2_acc, 'ro-', linewidth=lw, label="w/ accelerometer")
axs[2].set_title('Relative Acceleration $\mathbf{Y}_2$')
axs[2].set_xlabel('SNR [dB]')
axs[2].set_ylabel('RMSE [$m/s^2$]')
plt.setp(axs[2], xticks=[0, 5, 10, 15, 20, 25, 30])
plt.setp(axs[2], yticks=[1e-1, 1e-3, 1e-5])
# axs[2].legend()
axs[2].grid()
fig.tight_layout(h_pad=-0.2)

plt.show()

print('finished!')