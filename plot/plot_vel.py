import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import norm
import utils_rl

mpl.use('Qt5Agg')
# font = {'size': 12}
# mpl.rc('font', **font)
plt.ion()

crlb = np.load('./../output/crlb_vel.npz')
mc_runs = np.load('./../output/comp_vel.npz')

Y0 = np.array([[-244.0, -588.0], [385.0, -456.0], [81.0, -992.0], [-19.0, -730.0], [-792.0, 879.0], [-554.0, 970.0],
                   [-965.0, 155.0], [-985.0, 318.0], [-49.0, -858.0], [-503.0, 419.0]]).transpose()
Y1 = np.array(
        [[-5.0, -8.0], [-8.0, -5.0], [-6.0, -7.0], [6.0, -9.0], [-1.0, -3.0], [2.0, -2.0], [1.0, -2.0], [-5.0, -10.0],
         [9.0, 2.0], [-5.0, -1.0]]).transpose()
tend = 5.0
# dimensions
nDim, N = Y0.shape
N_bar = int(N * (N - 1) / 2.)
n_bar = int(N * (N + 1) / 2.)
one = np.ones((N, 1))
C = np.eye(N) - (one @ one.T) / N
Y0_bar = Y0 @ C
Y1_bar = Y1 @ C

crlb_K = crlb['K']
crlb_main_y0 = crlb['rmse_main_y0']
crlb_main_y1 = crlb['rmse_main_y1']
crlb_main_b0 = crlb['rmse_main_b0']
crlb_main_b1 = crlb['rmse_main_b1']
crlb_main_b2 = crlb['rmse_main_b2']
crlb_gtwr_y0 = crlb['rmse_gtwr_y0']
crlb_gtwr_y1 = crlb['rmse_gtwr_y1']

rmse_K = mc_runs['K']
rmse_main_y0 = mc_runs['rmse_main_y0']
rmse_main_y1 = mc_runs['rmse_main_y1']
rmse_main_b0 = mc_runs['rmse_main_b0']
rmse_main_b1 = mc_runs['rmse_main_b1']
rmse_main_b2 = mc_runs['rmse_main_b2']
rmse_gtwr_y0 = mc_runs['rmse_gtwr_y0']
rmse_gtwr_y1 = mc_runs['rmse_gtwr_y1']
rmse_gtwr_b0 = mc_runs['rmse_gtwr_b0']
rmse_gtwr_b1 = mc_runs['rmse_gtwr_b1']
rmse_gtwr_b2 = mc_runs['rmse_gtwr_b2']
err_main_y0 = mc_runs['err_main_y0']
err_main_y1 = mc_runs['err_main_y1']
err_gtwr_y0 = mc_runs['err_gtwr_y0']
err_gtwr_y1 = mc_runs['err_gtwr_y1']

# centereted trajectory
N_EXP = mc_runs['Y0_main_bar'].shape[2]
Y0_main = mc_runs['Y0_main_bar']
Y1_main_tilde = mc_runs['Y1_main_tilde']
H1_main = mc_runs['H1_main']
rmse_X = [None] * len(rmse_K)
for kk in range(len(rmse_K)):
    K = rmse_K[kk]
    X_bar = np.zeros((nDim, N, K + 1))
    X_est = np.zeros((nDim, N, N_EXP, K + 1))
    X_est_align = np.zeros((nDim, N, N_EXP, K + 1))
    t_pwd = np.linspace(-tend, tend, K + 1).reshape(K + 1, 1)
    idx = np.argmin(np.abs(t_pwd))
    err_X = np.zeros((nDim * N, N_EXP, K + 1))
    rmse_X[kk] = np.zeros(K + 1)

    # Setting the trajectory for each time slot
    for ii in range(K + 1):
        X_bar[:, :, ii] = Y0_bar + Y1_bar * t_pwd[ii]

    # calculating the trajectory estimate
    for jj in range(N_EXP):
        for ii in range(K + 1):
            X_bar[:, :, ii] = Y0_bar + Y1_bar * t_pwd[ii]
            Y1_main = H1_main[:, :, jj, kk] @ Y1_main_tilde[:, :, jj, kk]
            X_est[:, :, jj, ii] = Y0_main[:, :, jj, kk] + Y1_main * t_pwd[ii]

        # The two trajectories are not aligned. Thus, we do the alignment at t = 0!
        err, H = utils_rl.procrustes_error(X_est[:, :, jj, idx], Y0_bar)
        # calculating the error per time step
        for ii in range(K + 1):
            X_est_align[:, :, jj, ii] = H.T @ X_est[:, :, jj, ii]
            err_X[:, jj, ii] = np.squeeze(utils_rl.vectorize(X_bar[:, :, ii] - X_est_align[:, :, jj, ii]))

    # calculating the rmse per time step combining all runs
    rmse_X[kk] = np.sqrt(np.sum(norm(err_X, axis=0) ** 2, axis=0) / N_EXP) / (nDim * N)

lw = 3
fig, axs = plt.subplots(2, 1)
axs[0].set_yscale('log')
axs[1].set_yscale('log')

axs[0].plot(rmse_K, rmse_gtwr_y0, 'ro-', linewidth=lw, label="SOTA [27]")
axs[0].plot(rmse_K, rmse_main_y0, 'bo-', linewidth=lw, label="proposed")
axs[0].plot(crlb_K, crlb_gtwr_y0, 'rx--', linewidth=lw, label="CRLB [27]")
axs[0].plot(crlb_K, crlb_main_y0, 'bx--', linewidth=lw, label="CRLB proposed")
plt.setp(axs[0], yticks=[1e-2, 8e-3, 6e-3, 4e-3, 2e-3, 1e-3])
# plt.setp(axs[0], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs[0].set_ylabel('RMSE [$m$]')
axs[0].set_title('Relative Position $\mathbf{Y}_0$', fontsize=12)
# axs[0].set_xticks([])
axs[0].xaxis.set_ticklabels([])
axs[0].legend()
axs[0].grid()

axs[1].plot(rmse_K, rmse_gtwr_y1, 'ro-', linewidth=lw, label="SOTA [27]")
axs[1].plot(rmse_K, rmse_main_y1, 'bo-', linewidth=lw, label="proposed")
axs[1].plot(crlb_K, crlb_gtwr_y1, 'rx--', linewidth=lw, label="CRLB [27]")
axs[1].plot(crlb_K, crlb_main_y1, 'bx--', linewidth=lw, label="CRLB proposed")
axs[1].set_ylabel('RMSE [$m/s$]')
axs[1].set_title('Relative Velocity $\mathbf{Y}_1$', fontsize=12)
plt.setp(axs[1], yticks=[2e-1, 1e-1, 8e-2, 6e-2, 4e-2, 2e-2, 1e-2])
# plt.setp(axs[1], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs[1].set_xlabel('Number of timestamps, $K$ [-]')
# axs[1].legend()
axs[1].grid()

fig4, axs4 = plt.subplots(1, 1)
axs4.set_yscale('log')
# for ii in range(len(rmse_X)):
#     K = rmse_K[ii]
#     axs4.plot(np.linspace(-tend, tend, K + 1), rmse_X[ii], 'ro-', linewidth=lw, label="K = " + str(rmse_K[ii]))
axs4.plot(np.linspace(-tend, tend, rmse_K[0] + 1), rmse_X[0], 'bs-', lw=lw, ms=10., label="K = " + str(rmse_K[0]))
axs4.plot(np.linspace(-tend, tend, rmse_K[2] + 1), rmse_X[2], 'r^--', lw=lw, ms=10., label="K = " + str(rmse_K[2]))
axs4.plot(np.linspace(-tend, tend, rmse_K[5] + 1), rmse_X[5], 'g.-.', lw=lw, ms=10., label="K = " + str(rmse_K[5]))
axs4.plot(np.linspace(-tend, tend, rmse_K[0] + 1), 0.01 * np.ones(rmse_K[0] + 1), 'k.--', linewidth=lw, label="$\sigma_d$ = " + str(0.01) + ' m')

# plt.setp(axs4[0], yticks=[1e-2, 8e-3, 6e-3, 4e-3, 2e-3, 1e-3])
# plt.setp(axs[0], xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axs4.set_xlabel('time [s]')
axs4.set_ylabel('RMSE $\mathbf{X}(t)$ [$m$]')
axs4.set_title('Relative trajectory error', fontsize=12)
# axs[0].set_xticks([])
# axs4[0].xaxis.set_ticklabels([])
axs4.legend()
axs4.grid()

plt.show()

print('finished!')

