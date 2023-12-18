import numpy as np
from scipy.linalg import null_space
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import utils_rl as utils
import math
import matplotlib as mpl
mpl.use('Qt5Agg')

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
THETA = np.pi / 3.

def jac(v, t, x0):
    x = x0 + v * t
    f = np.zeros((len(t)))
    J = np.zeros(x.shape)
    for ss in range(len(t)):
        f[ss] = np.sqrt(x[ss, :].dot(x[ss, :].transpose()))
        J[ss, :] = (x0 * t[ss] + v * t[ss] ** 2) / f[ss]

    return J

if __name__ == "__main__":
    import pathlib

    save_flag = True
    STD_ARRAY = [1., 0.1, 0.01, 0.001]

    '''
        Trajectory generation
    '''
    # # from Anu: modification from Raj's case so as to not have the constraint that the first two nodes
    # # are relatively static
    Y0 = np.array([[-244.0, -588.0], [385.0, -456.0], [81.0, -992.0], [-19.0, -730.0], [-792.0, 879.0], [-554.0, 970.0],
                   [-965.0, 155.0], [-985.0, 318.0], [-49.0, -858.0], [-503.0, 419.0]]).transpose()
    Y1 = np.array(
        [[-5.0, -8.0], [-8.0, -5.0], [-6.0, -7.0], [6.0, -9.0], [-1.0, -3.0], [2.0, -2.0], [1.0, -2.0], [-5.0, -10.0],
         [9.0, 2.0], [-5.0, -1.0]]).transpose()
    Y2 = np.array(
        [[-0.17, 0.42], [-0.42, 0.17], [0.22, 0.98], [-0.07, 0.73], [0.21, 0.48], [-0.15, 0.08], [0.55, -0.43],
         [-0.72, -0.14], [-0.49, 0.56], [-0.34, 0.91]]).transpose()

    # from Anu: modification from Raj's case so as to not have the constraint that the first two nodes
    # are relatively static
    # X0 = np.array([[-24.0, -58.0], [35.0, -45.0], [81.0, -29.0], [-19.0, -30.0], [-27.0, 79.0], [-54.0, 70.0],
    #                [-65.0, 15.0], [-45.0, 18.0], [-49.0, -58.0], [-50.0, 19.0]]).transpose()
    # Y1 = np.array(
    #     [[-2.0, -5.0], [-5.0, -2.0], [-3.0, -4.0], [3.0, -6.0], [-1.0, -3.0], [2.0, -2.0], [1.0, -2.0], [-2.0, -1.0],
    #      [4.0, 2.0], [-5.0, -1.0]]).transpose()

    # parameters
    K_array = np.array([10, 30, 50, 70, 90, 100])
    dt = 0.01
    tend = 5.0

    # dimensions
    nDim, N = Y0.shape
    N_bar = int(N * (N - 1) / 2.)
    n_bar = int(N * (N + 1) / 2.)
    n_Acc = int(N * nDim)
    d_bar = 2
    one = np.ones((N, 1))
    C = np.eye(N) - (one @ one.T) / N
    Y0_bar = Y0 @ C
    Y1_bar = Y1 @ C
    Y2_bar = Y2 @ C
    B0 = Y0_bar.T @ Y0_bar
    b0 = utils.half_vectorize(B0)
    B1 = Y0_bar.T @ Y1_bar + Y1_bar.T @ Y0_bar
    b1 = utils.half_vectorize(B1)
    B2 = Y1_bar.T @ Y1_bar
    b2 = utils.half_vectorize(B2)
    B3 = 0.5 * (Y1_bar.T @ Y2_bar + Y2_bar.T @ Y1_bar)
    b3 = utils.half_vectorize(B3)
    B4 = 0.25 * (Y2_bar.T @ Y2_bar)
    b4 = utils.half_vectorize(B4)

    # Let the acceleration values be in the unknown frame of the mobile node
    H2 = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])
    Y2_tilde = H2 @ Y2_bar

    # jacobian for relative position
    idx = np.triu_indices(N)
    J0_main = np.zeros((n_bar, N * nDim))
    for ii in range(n_bar):  # # going through all possible pairwise distance indices
        for jj in range(N):  # for each node position
            if idx[0][ii] == idx[1][ii] and jj == idx[0][ii]:
                J0_main[ii, jj * nDim: (jj + 1) * nDim] = 2. * (Y0_bar[:, idx[0][ii]])
            elif jj == idx[0][ii]:
                J0_main[ii, jj * nDim: (jj + 1) * nDim] = Y0_bar[:, idx[1][ii]]
            elif jj == idx[1][ii]:
                J0_main[ii, jj * nDim: (jj + 1) * nDim] = Y0_bar[:, idx[0][ii]]
            else:
                J0_main[ii, jj * nDim: (jj + 1) * nDim] = np.zeros(2)

    # jacobians for relative velocity and orientation
    r = np.zeros((2 * n_bar, 1))
    for ii in range(2 * n_bar):
        if ii < n_bar:
            r[ii] = Y0_bar[:, idx[0][ii]].T @ Y1_bar[:, idx[1][ii]] + Y1_bar[:, idx[0][ii]].T @ Y0_bar[:, idx[1][ii]]
        else:
            jj = ii - n_bar
            r[ii] = Y2_bar[:, idx[0][jj]].T @ Y1_bar[:, idx[1][jj]] + Y1_bar[:, idx[1][jj]].T @ Y2_bar[:, idx[1][jj]]
    r_theta = np.concatenate((b1, 2 * b3))

    J1_main = np.zeros((2 * n_bar, n_Acc + d_bar))
    i = idx[0]
    j = idx[1]
    for ii in range(2 * n_bar): # going through all possible pairwise distance indices
        if ii < n_bar:
            for kk in range(N): # for each node position
                if kk == i[ii] and kk == j[ii]:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = 2 * Y0_bar[:, j[ii]]
                elif kk == i[ii]:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = Y0_bar[:, j[ii]]
                elif kk == j[ii]:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = Y0_bar[:, i[ii]]
                else:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = np.zeros(nDim)
        else:
            ll = ii - n_bar # derivative w.r.t.orientation
            J1_main[ii, -2] = Y2_tilde[:, i[ll]].T @ Y0_bar[:, j[ll]] + Y2_tilde[:, j[ll]].T @ Y0_bar[:, i[ll]]
            ci_tilde = np.array([[-1, 0], [0, 1]]) @ Y2_tilde[:, i[ll]]
            cj_tilde = np.array([[-1, 0], [0, 1]]) @ Y2_tilde[:, j[ll]]
            J1_main[ii, -1] = ci_tilde.T @ Y0_bar[:, j[ll]] + cj_tilde.T @ Y0_bar[:, i[ll]]

            for kk in range(N):
                if kk == i[ll] and kk == j[ll]:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = 2 * Y2_tilde[:, j[ll]].T @ H2
                elif kk == i[ll]:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = Y2_tilde[:, j[ll]].T @ H2 # H2'??
                elif kk == j[ll]:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = Y2_tilde[:, i[ll]].T @ H2
                else:
                    J1_main[ii, kk * nDim: (kk + 1) * nDim] = np.zeros(nDim)

    J2_main = np.zeros((n_bar, N * nDim))
    for ii in range(n_bar):  # # going through all possible pairwise distance indices
        for jj in range(N):  # for each node position
            if idx[0][ii] == idx[1][ii] and jj == idx[0][ii]:
                J2_main[ii, jj * nDim: (jj + 1) * nDim] = 2. * (Y1_bar[:, idx[0][ii]])
            elif jj == idx[0][ii]:  # jj == idx[0][ii]:
                J2_main[ii, jj * nDim: (jj + 1) * nDim] = Y1_bar[:, idx[1][ii]]
            elif jj == idx[1][ii]:  # jj == idx[1][ii]:
                J2_main[ii, jj * nDim: (jj + 1) * nDim] = Y1_bar[:, idx[0][ii]]
            else:
                J2_main[ii, jj * nDim: (jj + 1) * nDim] = np.zeros(nDim)


    Dm = utils.duplication_matrix_char(N)
    M = -0.5 * pinv(Dm) @ np.kron(C.T, C) @ Dm

    Sigma_b0 = np.zeros((n_bar, n_bar, len(K_array), len(STD_ARRAY)))
    Sigma_b1 = np.zeros((n_bar, n_bar, len(K_array), len(STD_ARRAY)))
    Sigma_b2 = np.zeros((n_bar, n_bar, len(K_array), len(STD_ARRAY)))
    Sigma_b3 = np.zeros((n_bar, n_bar, len(K_array), len(STD_ARRAY)))
    Sigma_b4 = np.zeros((n_bar, n_bar, len(K_array), len(STD_ARRAY)))
    rmse_main_y0 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_y1 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_y1_ctr = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_y2 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_b0 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_b1 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_b2 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_b3 = np.zeros((len(K_array), len(STD_ARRAY)))
    rmse_main_b4 = np.zeros((len(K_array), len(STD_ARRAY)))

    for kk in range(len(STD_ARRAY)):
        STD_DIST = STD_ARRAY[kk]
        for mm in range(len(K_array)):
            K = K_array[mm]
            print([mm, K])
            t_pwd = np.linspace(-tend, tend, K + 1).reshape(K + 1, 1)

            # Forming matrix T
            I_nbar = np.identity(n_bar)
            T0 = np.kron(np.ones(t_pwd.shape), I_nbar)
            T1 = np.kron(t_pwd, I_nbar)
            T2 = np.kron(np.square(t_pwd), I_nbar)
            T3 = np.kron(np.power(t_pwd, 3), I_nbar)
            T4 = np.kron(np.power(t_pwd, 4), I_nbar)
            T = np.hstack((T0, T1, T2, T3, T4))

            Sigma_d = STD_DIST ** 2 * np.identity(n_bar)
            Sigma_g = np.zeros((n_bar * (K + 1), n_bar * (K + 1)))

            for ii in range(K + 1):
                X = Y0 + Y1 * t_pwd[ii]
                # d = utils.half_vectorize(utils.pairwise_distance(X), skew=True).reshape(N_bar)
                d = utils.half_vectorize(utils.pairwise_distance(X)).reshape(n_bar)
                Sigma_calD = 4. * np.diag(d) @ Sigma_d @ np.diag(d)
                Sigma_g[ii * n_bar: (ii + 1) * n_bar, ii * n_bar: (ii + 1) * n_bar] = M @ Sigma_calD @ M.T

            # covariance for b0, b1, b2
            fim_theta = T.T @ pinv(Sigma_g) @ T
            Sigma_theta = pinv(fim_theta)

            Sigma_b0[:, :, mm, kk] = Sigma_theta[:n_bar, :n_bar]
            Sigma_b1[:, :, mm, kk] = Sigma_theta[n_bar:2 * n_bar, n_bar:2 * n_bar]
            Sigma_b2[:, :, mm, kk] = Sigma_theta[2 * n_bar: 3 * n_bar, 2 * n_bar: 3 * n_bar]
            Sigma_b3[:, :, mm, kk] = Sigma_theta[3 * n_bar: 4 * n_bar, 3 * n_bar: 4 * n_bar]
            Sigma_b4[:, :, mm, kk] = Sigma_theta[4 * n_bar:, 4 * n_bar:]
            rmse_main_b0[mm, kk] = np.sqrt(np.trace(Sigma_b0[:, :, mm, kk])) / n_bar
            rmse_main_b1[mm, kk] = np.sqrt(np.trace(Sigma_b1[:, :, mm, kk])) / n_bar
            rmse_main_b2[mm, kk] = np.sqrt(np.trace(Sigma_b2[:, :, mm, kk])) / n_bar
            rmse_main_b3[mm, kk] = np.sqrt(np.trace(Sigma_b3[:, :, mm, kk])) / n_bar
            rmse_main_b4[mm, kk] = np.sqrt(np.trace(Sigma_b4[:, :, mm, kk])) / n_bar

            # CRLB for relative position estimate
            fim_main_y0 = J0_main.T @ pinv(Sigma_b0[:, :, mm, kk]) @ J0_main
            crlb_main_y0 = pinv(fim_main_y0)
            rmse_main_y0[mm, kk] = np.sqrt(np.trace(crlb_main_y0)) / (N * nDim)

            # CRLB for relative velocity estimate
            SigmaY1 = block_diag(Sigma_b1[:, :, mm, kk], 4 * Sigma_b3[:, :, mm, kk])
            fim_main_y1 = J1_main.T @ pinv(SigmaY1) @ J1_main
            crlb_main_y1 = pinv(fim_main_y1)
            rmse_main_y1[mm, kk] = np.sqrt(np.trace(crlb_main_y1[:n_Acc, :n_Acc])) / (N * nDim)

            # constraint matrix:
            F = np.zeros((1, n_Acc + 2))
            F[0, -2] = 2 * H2[0, 0]
            F[0, -1] = 2 * H2[0, 1]

            U = null_space(F)
            crlb_main_y1_ctr = U @ pinv(U.T @ fim_main_y1 @ U) @ U.T
            rmse_main_y1_ctr[mm, kk] = np.sqrt(np.trace(crlb_main_y1_ctr[:n_Acc, :n_Acc])) / (N * nDim)

            print([rmse_main_y1[mm, kk], rmse_main_y1_ctr[mm, kk]])
            # CRLB for relative acceleration estimate
            fim_main_y2 = J2_main.T @ pinv(16 * Sigma_b4[:, :, mm, kk]) @ J2_main
            crlb_main_y2 = pinv(fim_main_y2)
            rmse_main_y2[mm, kk] = np.sqrt(np.trace(crlb_main_y2)) / (N * nDim)

    # plots
    fig, axs = plt.subplots(3, 1)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    axs[0].plot(K_array, rmse_main_y0[:, 2], 'bo-', label='proposed')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(K_array, rmse_main_y1[:, 2], 'bo-', label='proposed')
    axs[1].plot(K_array, rmse_main_y1_ctr[:, 2], 'r.--', label='constrained')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(K_array, rmse_main_y2[:, 2], 'bo-', label='proposed')
    axs[2].grid(True)
    axs[2].legend()

    fig2, axs2 = plt.subplots(1, 1)
    axs2.set_yscale('log')
    cline = ['b', 'r', 'm', 'k']
    for kk in range(len(STD_ARRAY)):
        axs2.plot(K_array, rmse_main_y1[:, kk], 'o-', c=cline[kk], label='unconstrained')
        axs2.plot(K_array, rmse_main_y1_ctr[:, kk], '.--', c=cline[kk], label='constrained')
    axs2.grid(True)
    axs2.legend()

    plt.show()

    if save_flag:
        out_name = 'crlb_acc3'
        output = pathlib.Path(out_name)
        results_path = output.with_suffix('.npz')
        with open(results_path, 'xb') as results_file:
            np.savez(results_file, N=N, K=K_array, rmse_main_y0=rmse_main_y0, rmse_main_y1=rmse_main_y1, rmse_main_y2=rmse_main_y2,
                         rmse_main_y1_ctr=rmse_main_y1_ctr, rmse_main_b0=rmse_main_b0, rmse_main_b1=rmse_main_b1, rmse_main_b2=rmse_main_b2,
                         rmse_main_b3=rmse_main_b3, rmse_main_b4=rmse_main_b4, std_array=STD_ARRAY)

    print('finished!')