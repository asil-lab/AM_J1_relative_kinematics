import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils_rl as utils
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

# Y0 = Y0[:, :5]
# Y1 = Y1[:, :5]
# Y2 = Y2[:, :5]

H = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])

lle = np.load('./main_lle_N10.npz')
# lle = np.load('./main_lle_N5.npz')
# lle = np.load('./main_lle_2.npz')
lle_ctr = np.load('./main_lle_ctr_N10.npz')
# lle_ctr = np.load('./main_lle_ctr_N5.npz')

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

fig6, axs6 = plt.subplots(M, 1)
for ii in range(M):
    axs6[ii].set_yscale('log')
    axs6[ii].plot(-SNR_B1, corr_rmse_yd_hat[:, ii], 'bo-', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
    axs6[ii].plot(-SNR_B1_ctr, corr_rmse_yd_hat_ctr[:, ii], 'rs--', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
    axs6[ii].set_ylabel('rmse [$m/s$]')
    axs6[ii].legend()
    axs6[ii].grid()
plt.show()
# removing outl

fig3, axs3 = plt.subplots(N, M-1)
idx = 0
for ii in range(N):
    for jj in range(M-1):
        axs3[ii, jj].plot(err_y_hat[idx, :, ii, jj])
        # axs3[ii, jj].legend()

cols = ["SNR $B_3$ (dB) = {}".format(col) for col in SNR_B3]
rows = ["SNR $B_1$ (dB) = {}".format(row) for row in SNR_B1]
for ax, col in zip(axs3[0, :-1], cols):
    ax.set_title(col)

for ax, row in zip(axs3[:, 0], rows):
    ax.set_ylabel(row, rotation=0)

# fig3.tight_layout()
plt.show()

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

fig7, axs7 = plt.subplots(N, M-1)
idx = 0
for ii in range(N):
    for jj in range(M-1):
        axs7[ii, jj].plot(theta_hat[:, ii, jj])
        axs7[ii, jj].plot(THETA * np.ones(N_EXP), 'grey')
        # axs3[ii, jj].legend()

cols = ["SNR $B_3$ (dB) = {}".format(col) for col in SNR_B3]
rows = ["SNR $B_1$ (dB) = {}".format(row) for row in SNR_B1]
for ax, col in zip(axs7[0, :], cols):
    ax.set_title(col)

for ax, row in zip(axs7[:, 0], rows):
    ax.set_ylabel(row, rotation=0)

# fig3.tight_layout()
plt.show()

# fig5, axs5 = plt.subplots(M, 1)
# for ii in range(M):
#     axs5[ii].set_yscale('log')
#     axs5[ii].plot(-SNR_B1, rmse_theta_hat[:, ii], 'bo-', linewidth=lw, label="SNR $\mathbf{B}_3$ $=-$ " + str(SNR_B3[ii]) + " dB")
#     axs5[ii].plot(-SNR_B1_ctr, rmse_theta_hat_ctr[:, ii], 'rs--', linewidth=lw, label="SNR $\mathbf{B}_3$ $=-$ " + str(SNR_B3[ii]) + " dB")
#     axs5[ii].set_ylabel('rmse [$rad$]')
#     axs5[ii].legend()
#     axs5[ii].grid()
# plt.show()

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

# H estimate - constrained and unconstrained
# fig4, axs4 = plt.subplots(M, 1)
# for ii in range(M):
#     axs4[ii].set_yscale('log')
#     axs4[ii].plot(-SNR_B1, rmse_h_hat[:, ii], 'bo-', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs4[ii].plot(-SNR_B1_ctr, rmse_h_hat_ctr[:, ii], 'rs--', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs4[ii].set_ylabel('rmse [$m/s$]')
#     axs4[ii].legend()
#     axs4[ii].grid()
# plt.show()

# id_to_plot = [0, 1, 2, 3, 4, 5]
# fig2, axs2 = plt.subplots(M, 1)
# for ii in range(err_y_hat.shape[0]):
#     for jj in range(M):
#         axs2[jj].clear()
#         axs2[jj].plot(err_y_hat[ii, :, 0, jj], 'bo', label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[jj]) + " dB")
#         axs2[jj].legend()
#         axs2[jj].grid()
#         fig2.suptitle('[' + str(ii) + ', ' + str(SNR_B1[0]) + " dB, " + str(SNR_B3[jj]) + ' dB]')
#     plt.draw()
#     plt.pause(10)

# plt.figure()
# idxs = [0, -1, -1]
# plt.plot(err_y_hat[idxs[0], :, idxs[1], idxs[2]], 'bo')
# plt.title('[' + str(idxs[0]) + ', SNR B1 = ' + str(SNR_B1[idxs[1]]) + " dB, SNR B3 = " + str(SNR_B3[idxs[-1]]) + ' dB]')
# plt.grid()
# plt.show()

# fig2, axs2 = plt.subplots(M, 1)

# for ii in range(err_y_hat.shape[0]):
#     for jj in range(M):
#         axs2[jj].clear()
#         axs2[jj].plot(err_y_hat[ii, :, 0, jj], 'bo', label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[jj]) + " dB")
#         axs2[jj].legend()
#         axs2[jj].grid()
#         fig2.suptitle('[' + str(ii) + ', ' + str(SNR_B1[0]) + " dB, " + str(SNR_B3[jj]) + ' dB]')
#     plt.draw()
#     plt.pause(10)

# for ii in range(M):
#     axs[ii].set_yscale('log')
#     axs[ii].plot(-SNR_B1, rmse_y_hat[:, ii], 'bo-', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs[ii].plot(-SNR_B1_ctr, rmse_y_hat_ctr[:, ii], 'rs--', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs[ii].set_ylabel('rmse [$m/s$]')
#     axs[ii].legend()
#     axs[ii].grid()
# plt.show()

# for ii in range(4):
#     axs[ii].set_yscale('log')
#     axs[ii].plot(-SNR_B1, rmse_y2_hat[:, ii], 'bo-', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs[ii].plot(-SNR_B1_ctr, rmse_y2_hat_ctr[:, ii], 'rs--', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs[ii].set_ylabel('rmse [$m/s$]')
#     axs[ii].legend()
#     axs[ii].grid()
# plt.show()

# for ii in range(4):
#     axs[ii].set_yscale('log')
#     axs[ii].plot(-SNR_B1, rmse_yd_hat[:, ii], 'bo-', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs[ii].plot(-SNR_B1_ctr, rmse_yd_hat_ctr[:, ii], 'rs--', linewidth=lw, label="SNR $\mathbf{B}_3$ $=$ " + str(SNR_B3[ii]) + " dB")
#     axs[ii].set_ylabel('rmse [$m/s$]')
#     axs[ii].legend()
#     axs[ii].grid()
# plt.show()

fig3, axs3 = plt.subplots(3, 1)
axs3[0].set_yscale('log')
axs3[1].set_yscale('log')
axs3[2].set_yscale('log')

axs3[0].plot(rmse_K, rmse_y0_hat, 'bo-', linewidth=lw, label="proposed")
axs3[0].plot(rmse_K, crlb_main_y0, 'kx--', linewidth=lw, label="CRLB")
axs3[0].set_ylabel('rmse [$m$]')
axs3[0].set_title('Relative Position $\mathbf{Y}_0$', fontsize=12)
plt.setp(axs3[0], yticks=[1e-3, 1e-4])
plt.setp(axs3[0], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs3[0].xaxis.set_ticklabels([])
# axs[0]3.set_xlabel('No. of timestamps [K]')
axs3[0].legend()
axs3[0].grid()

axs3[1].plot(rmse_K, rmse_y1_hat, 'bo-', linewidth=lw, label="proposed")
axs3[1].plot(rmse_K, crlb_main_y1, 'kx--', linewidth=lw, label="CRLB")
axs3[1].set_ylabel('rmse [$m/s$]')
axs3[1].set_title('Relative Velocity $\mathbf{Y}_1$', fontsize=12)
plt.setp(axs3[1], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs3[1].xaxis.set_ticklabels([])
axs3[1].legend()
axs3[1].grid()

axs3[2].plot(rmse_K, rmse_y2_hat, 'bo-', linewidth=lw, label="proposed")
axs3[2].plot(rmse_K, crlb_main_y2, 'kx--', linewidth=lw, label="CRLB")
axs3[2].set_ylabel('rmse [$m/s^2$]')
axs3[2].set_title('Relative Acceleration $\mathbf{Y}_2$', fontsize=12)
axs3[2].set_xlabel('No. of timestamps [K]')
plt.setp(axs3[2], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs3[2].legend()
axs3[2].grid()

fig2, axs2 = plt.subplots(5, 1)
axs2[0].set_yscale('log')
axs2[1].set_yscale('log')
axs2[2].set_yscale('log')
axs2[3].set_yscale('log')
axs2[4].set_yscale('log')

axs2[0].plot(rmse_K, rmse_b0_hat, 'bo-')
axs2[0].plot(rmse_K, rmse_b0_acc, 'rs-')
axs2[0].grid()

axs2[1].plot(rmse_K, rmse_b1_hat, 'bo-')
axs2[1].plot(rmse_K, rmse_b1_acc, 'rs-')
axs2[1].grid()

axs2[2].plot(rmse_K, rmse_b2_hat, 'bo-')
axs2[2].plot(rmse_K, rmse_b2_acc, 'rs-')
axs2[2].grid()

axs2[3].plot(rmse_K, rmse_b3_hat, 'bo-')
axs2[3].plot(rmse_K, rmse_b3_acc, 'rs-')
axs2[3].grid()

axs2[4].plot(rmse_K, rmse_b4_hat, 'bo-')
axs2[4].grid()

plt.show()

print('finished!')

