import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
from scipy.linalg import orthogonal_procrustes
from numpy.linalg import norm
mpl.use('Qt5Agg')
font = {'size': 12}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]
plt.ion()
THETA = np.pi / 3.

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

Y0 = np.array([[-244.0, -588.0], [385.0, -456.0], [81.0, -992.0], [-19.0, -730.0], [-792.0, 879.0], [-554.0, 970.0],
                   [-965.0, 155.0], [-985.0, 318.0], [-49.0, -858.0], [-503.0, 419.0]]).transpose()
Y1 = np.array([[-5.0, -8.0], [-8.0, -5.0], [-6.0, -7.0], [6.0, -9.0], [-1.0, -3.0], [2.0, -2.0], [1.0, -2.0], [-5.0, -10.0],
         [9.0, 2.0], [-5.0, -1.0]]).transpose()
Y2 = np.array([[-0.17, 0.42], [-0.42, 0.17], [0.22, 0.98], [-0.07, 0.73], [0.21, 0.48], [-0.15, 0.08], [0.55, -0.43],
         [-0.72, -0.14], [-0.49, 0.56], [-0.34, 0.91]]).transpose()

H = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])

lle = np.load('./../output/main_lle_N10.npz')
lle_ctr = np.load('./../output/main_lle_ctr_N10.npz')

Y_hat = lle['Y_hat']
err_y_hat = lle['err_y_hat']
err_y2_hat = lle['err_y2_hat']
err_yd_hat = lle['err_yd_hat']
err_ynd_hat = lle['err_ynd_hat']
rmse_y_hat = lle['rmse_y_hat']
rmse_y2_hat = lle['rmse_y2_hat']
rmse_yd_hat = lle['rmse_yd_hat']
rmse_ynd_hat = lle['rmse_ynd_hat']
H_hat = lle['H_hat']
err_h_hat = lle['err_h_hat']
rmse_h_hat = lle['rmse_h_hat']
SNR_B1 = lle['SNR_B1']
SNR_B3 = lle['SNR_B3']

Y_hat_ctr = lle_ctr['Y_hat']
err_y_hat_ctr = lle_ctr['err_y_hat']
err_y2_hat_ctr = lle_ctr['err_y2_hat']
err_yd_hat_ctr = lle_ctr['err_yd_hat']
err_ynd_hat_ctr = lle_ctr['err_ynd_hat']
rmse_y_hat_ctr = lle_ctr['rmse_y_hat']
rmse_y2_hat_ctr = lle_ctr['rmse_y2_hat']
rmse_yd_hat_ctr = lle_ctr['rmse_yd_hat']
rmse_ynd_hat_ctr = lle_ctr['rmse_ynd_hat']
H_hat_ctr = lle_ctr['H_hat']
err_h_hat_ctr = lle_ctr['err_h_hat']
rmse_h_hat_ctr = lle_ctr['rmse_h_hat']
SNR_B1_ctr = lle_ctr['SNR_B1']
SNR_B3_ctr = lle_ctr['SNR_B3']

N = len(SNR_B1)
M = len(SNR_B3)
lw = 3.0

# recalculating err_ynd_hat properly
N_EXP = err_y_hat.shape[1]
corr_err_ynd_hat = np.zeros((2, N_EXP, N, M))
corr_err_ynd_hat_ctr = np.zeros((2, N_EXP, N, M))
corr_err_yd_hat = np.zeros((err_y_hat.shape[0] - 2, N_EXP, N, M))
corr_err_yd_hat_ctr = np.zeros((err_y_hat.shape[0] - 2, N_EXP, N, M))
corr_rmse_ynd_hat = np.zeros((N, M))
corr_rmse_ynd_hat_ctr = np.zeros((N, M))
corr_rmse_yd_hat = np.zeros((N, M))
corr_rmse_yd_hat_ctr = np.zeros((N, M))

for kk in range(N):
    for jj in range(M):
        for nn in range(N_EXP):
            # contains the off diagonal elements involved in ls
            corr_err_ynd_hat[:, nn, kk, jj] = np.squeeze(np.vstack((err_y_hat[1, nn, kk, jj], err_y_hat[2, nn, kk, jj])))
            corr_err_ynd_hat_ctr[:, nn, kk, jj] = np.squeeze(np.vstack((err_y_hat_ctr[1, nn, kk, jj], err_y_hat_ctr[2, nn, kk, jj])))

            # contains everything apart from the off diagonal elements involved in ls
            tmp = np.squeeze(np.vstack((err_y_hat[0, nn, kk, jj], err_y_hat[3, nn, kk, jj])))
            corr_err_yd_hat[:, nn, kk, jj] = np.concatenate((tmp, err_y2_hat[:, nn, kk, jj]))

            tmp = np.squeeze(np.vstack((err_y_hat_ctr[0, nn, kk, jj], err_y_hat_ctr[3, nn, kk, jj])))
            corr_err_yd_hat_ctr[:, nn, kk, jj] = np.concatenate((tmp, err_y2_hat_ctr[:, nn, kk, jj]))

        corr_rmse_ynd_hat[kk, jj] = np.sqrt(np.sum(np.square(norm(corr_err_ynd_hat[:, :, kk, jj], axis=0))) / N_EXP) / corr_err_ynd_hat.shape[0]
        corr_rmse_ynd_hat_ctr[kk, jj] = np.sqrt(np.sum(np.square(norm(corr_err_ynd_hat_ctr[:, :, kk, jj], axis=0))) / N_EXP) / corr_err_ynd_hat_ctr.shape[0]
        corr_rmse_yd_hat[kk, jj] = np.sqrt(np.sum(np.square(norm(corr_err_yd_hat[:, :, kk, jj], axis=0))) / N_EXP) / corr_err_yd_hat.shape[0]
        corr_rmse_yd_hat_ctr[kk, jj] = np.sqrt(np.sum(np.square(norm(corr_err_yd_hat_ctr[:, :, kk, jj], axis=0))) / N_EXP) / corr_err_yd_hat_ctr.shape[0]

theta_hat = np.zeros((H_hat.shape[2:]))
err_theta_hat = np.zeros((H_hat.shape[2:]))
rmse_theta_hat = np.zeros((H_hat.shape[3:]))
theta_hat_ctr = np.zeros((H_hat.shape[2:]))
err_theta_hat_ctr = np.zeros((H_hat.shape[2:]))
rmse_theta_hat_ctr = np.zeros((H_hat.shape[3:]))
for jj in range(H_hat.shape[3]):
    for kk in range(H_hat.shape[4]):
        for ii in range(H_hat.shape[2]):
            theta_hat[ii, jj, kk] = np.arctan2(-H_hat[0, 1, ii, jj, kk], H_hat[0, 0, ii, jj, kk])
            theta_hat_ctr[ii, jj, kk] = np.arctan2(-H_hat_ctr[0, 1, ii, jj, kk], H_hat_ctr[0, 0, ii, jj, kk])
            err_theta_hat[ii, jj, kk] = theta_hat[ii, jj, kk] - THETA
            err_theta_hat_ctr[ii, jj, kk] = theta_hat_ctr[ii, jj, kk] - THETA
        rmse_theta_hat[jj, kk] = norm(err_theta_hat[:, jj, kk])
        rmse_theta_hat_ctr[jj, kk] = norm(err_theta_hat_ctr[:, jj, kk])

fig, axs = plt.subplots(M-1, 1)
axs[0].set_ylim([3e-4, 1e-2])
axs[1].set_ylim([3e-3, 2e-2])
axs[2].set_ylim([2e-2, 2e-1])
for ii in range(M-1):
    axs[ii].set_yscale('log')
    axs[ii].plot(-SNR_B1[1:], rmse_yd_hat[1:, ii], 'bo-', linewidth=lw, label="least-squares")
    axs[ii].plot(-SNR_B1_ctr[1:], rmse_yd_hat_ctr[1:, ii], 'rs--', linewidth=lw, label="constrained least squares")
    axs[ii].set_ylabel('RMSE [$m/s$]')
    # axs[ii].title.set_text("SNR $\mathbf{B}_3 =$ " + str(-SNR_B3[ii]) + " dB")
    axs[ii].annotate("SNR $\mathbf{B}_3 =$ " + str(-SNR_B3[ii]) + " dB", xy=(0.5, 0.9), xycoords='axes fraction', size=12, ha='right', va='top')
    axs[ii].tick_params(axis='y', which='minor', labelsize=9)
    axs[ii].grid()
    if ii < M - 2:
        axs[ii].xaxis.set_ticklabels([])
        axs[ii].minorticks_off()
    else:
        axs[ii].set_xlabel('SNR $\mathbf{B}_1$ [-]')
ttl = fig.suptitle(r"$\bm{\psi}_1$")
ttl.set_position([.5, 0.922])
axs[0].legend()
axs[2].minorticks_off()

plt.show()

print('finished!')

