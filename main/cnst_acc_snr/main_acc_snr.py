import numpy as np
import matplotlib.pyplot as plt
import math
import utils
from matplotlib.gridspec import GridSpec
from numpy.linalg import pinv, norm
from scipy.linalg import orthogonal_procrustes

N_EXP = 1000
SEED_START = 2110
THETA = np.pi / 3.

def procrustes_error(Z, Z_bar):
    H, scale = orthogonal_procrustes(Z.T, Z_bar.T)
    Z_proc = Z.T @ H
    err_z = utils.vectorize(Z_bar - Z_proc.T)

    return np.squeeze(err_z), H

if __name__ == "__main__":
    import pathlib
    NOISE = 1
    save_flag = True
    SNR_gains = np.array([-30, -20, -15, -10, 0])
    STD_ACC = 0.001

    # Trajectory generation
    Y0 = np.array([[-244.0, -588.0], [385.0, -456.0], [81.0, -992.0], [-19.0, -730.0], [-792.0, 879.0], [-554.0, 970.0],
                   [-965.0, 155.0], [-985.0, 318.0], [-49.0, -858.0], [-503.0, 419.0]]).transpose()
    Y1 = np.array(
        [[-5.0, -8.0], [-8.0, -5.0], [-6.0, -7.0], [6.0, -9.0], [-1.0, -3.0], [2.0, -2.0], [1.0, -2.0], [-5.0, -10.0],
         [9.0, 2.0], [-5.0, -1.0]]).transpose()
    Y2 = np.array(
        [[-0.17, 0.42], [-0.42, 0.17], [0.22, 0.98], [-0.07, 0.73], [0.21, 0.48], [-0.15, 0.08], [0.55, -0.43],
         [-0.72, -0.14], [-0.49, 0.56], [-0.34, 0.91]]).transpose()

    # parameters
    K = 100
    tend = 5.0

    # dimensions
    nDim, N = Y0.shape
    n_pwd = int(N * (N - 1) / 2.)
    n_bar = int(N * (N + 1) / 2.)
    n_ACC = int(N * nDim)
    one = np.ones((N, 1))
    C = np.eye(N) - (one @ one.T) / N
    Y0_bar = Y0 @ C
    Y1_bar = Y1 @ C
    Y2_bar = Y2 @ C
    B0 = Y0_bar.T @ Y0_bar
    b0 = utils.half_vectorize(B0)
    B1 = Y0_bar.T @ Y1_bar + Y1_bar.T @ Y0_bar
    b1 = utils.half_vectorize(B1)
    B2 = 0.5 * (Y0_bar.T @ Y2_bar + Y2_bar.T @ Y0_bar) + Y1_bar.T @ Y1_bar
    b2 = utils.half_vectorize(B2)
    B3 = 0.5 * (Y1_bar.T @ Y2_bar + Y2_bar.T @ Y1_bar)
    b3 = utils.half_vectorize(B3)
    B4 = 0.25 * Y2_bar.T @ Y2_bar
    b4 = utils.half_vectorize(B4)
    theta = np.vstack((b0, b1, b2, b3, b4))
    # Let the acceleration values be in the unknown frame of the mobile node
    H2 = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])

    Dm = utils.duplication_matrix_char(N)
    M = -0.5 * pinv(Dm) @ np.kron(C.T, C) @ Dm

    Y0_bar_hat = np.zeros((nDim, N, N_EXP, len(SNR_gains)))
    Y1_bar_hat = np.zeros((nDim, N, N_EXP, len(SNR_gains)))
    Y2_tilde_hat = np.zeros((nDim, N, N_EXP, len(SNR_gains)))
    H2_hat = np.zeros((nDim, nDim, N_EXP, len(SNR_gains)))
    err_b0_hat = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b1_hat = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b2_hat = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b3_hat = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b4_hat = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_y0_hat = np.zeros((nDim * N, N_EXP, len(SNR_gains)))
    err_y1_hat = np.zeros((nDim * N, N_EXP, len(SNR_gains)))
    err_y2_hat = np.zeros((nDim * N, N_EXP, len(SNR_gains)))
    rmse_b0_hat = np.zeros(len(SNR_gains))
    rmse_b1_hat = np.zeros(len(SNR_gains))
    rmse_b2_hat = np.zeros(len(SNR_gains))
    rmse_b3_hat = np.zeros(len(SNR_gains))
    rmse_b4_hat = np.zeros(len(SNR_gains))
    rmse_y0_hat = np.zeros(len(SNR_gains))
    rmse_y1_hat = np.zeros(len(SNR_gains))
    rmse_y2_hat = np.zeros(len(SNR_gains))

    Y0_bar_acc = np.zeros((nDim, N, N_EXP, len(SNR_gains)))
    Y1_bar_acc = np.zeros((nDim, N, N_EXP, len(SNR_gains)))
    Y2_tilde_acc = np.zeros((nDim, N, N_EXP, len(SNR_gains)))
    H2_acc = np.zeros((nDim, nDim, N_EXP, len(SNR_gains)))
    err_b0_acc = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b1_acc = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b2_acc = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_b3_acc = np.zeros((n_bar, N_EXP, len(SNR_gains)))
    err_y0_acc = np.zeros((nDim * N, N_EXP, len(SNR_gains)))
    err_y1_acc = np.zeros((nDim * N, N_EXP, len(SNR_gains)))
    err_y2_acc = np.zeros((nDim * N, N_EXP, len(SNR_gains)))
    rmse_b0_acc = np.zeros(len(SNR_gains))
    rmse_b1_acc = np.zeros(len(SNR_gains))
    rmse_b2_acc = np.zeros(len(SNR_gains))
    rmse_b3_acc = np.zeros(len(SNR_gains))
    rmse_y0_acc = np.zeros(len(SNR_gains))
    rmse_y1_acc = np.zeros(len(SNR_gains))
    rmse_y2_acc = np.zeros(len(SNR_gains))

    t_acc = np.linspace(-tend, tend, 1001).reshape(1001, 1)
    for kk in range(len(SNR_gains)):
        STD_DIST = 10. ** (SNR_gains[kk] / 10.)
        Sigma_d = STD_DIST ** 2 * np.identity(n_bar)

        print([kk, SNR_gains[kk]])
        t_pwd = np.linspace(-tend, tend, K + 1).reshape(K + 1, 1)

        # Simulation
        X = np.zeros((nDim, N, K + 1))
        pwd = np.zeros((N, N, K + 1))
        pwd_noise = np.zeros((N, N, K + 1))

        for ii in range(K + 1):
            X[:, :, ii] = Y0 + Y1 * t_pwd[ii] + 0.5 * Y2 * t_pwd[ii] ** 2
            pwd[:, :, ii] = utils.pairwise_distance(X[:, :, ii])

        for nn in range(N_EXP):
            print([kk, nn])
            np.random.seed(SEED_START + 10 * nn)
            noise_val_dist = NOISE * np.random.normal(0.0, STD_DIST, n_pwd * (K + 1))
            noise_val_acc = NOISE * np.random.normal(0.0, STD_ACC, n_ACC * len(t_acc))

            D = np.zeros((N, N, K + 1))
            G = np.zeros((N, N, K + 1))
            vecG = np.zeros((n_bar * (K + 1), 1))
            Sigma_g = np.zeros((n_bar * (K + 1), n_bar * (K + 1)))
            w = np.zeros((K + 1) * n_bar)
            for ii in range(K + 1):
                vec_eta_dist = noise_val_dist[ii * n_pwd: (ii + 1) * n_pwd]
                eta_dist = utils.half_vectorize_inverse(vec_eta_dist, skew=True)
                pwd_noise[:, :, ii] = pwd[:, :, ii] + eta_dist
                D[:, :, ii] = np.square(pwd_noise[:, :, ii])
                G[:, :, ii] = utils.double_center(D[:, :, ii])
                vecG[ii * n_bar: (ii + 1) * n_bar] = utils.half_vectorize(G[:, :, ii])
                d = utils.half_vectorize(pwd_noise[:, :, ii]).reshape(n_bar)
                Sigma_calD = 4. * np.diag(d) @ Sigma_d @ np.diag(d)
                Sigma_g[ii * n_bar: (ii + 1) * n_bar, ii * n_bar: (ii + 1) * n_bar] = M @ Sigma_calD @ M.T
                w[ii * n_bar: (ii + 1) * n_bar] = np.diag(Sigma_g[ii * n_bar: (ii + 1) * n_bar, ii * n_bar: (ii + 1) * n_bar])
            # for jj in range(len(w)):
            #     w[jj] = 1 / Sigma_g[jj, jj]

            I_nbar = np.eye(n_bar)
            T0 = np.kron(np.ones(t_pwd.shape), I_nbar)
            T1 = np.kron(t_pwd, I_nbar)
            T2 = np.kron(np.square(t_pwd), I_nbar)
            T3 = np.kron(np.power(t_pwd, 3), I_nbar)
            T4 = np.kron(np.power(t_pwd, 4), I_nbar)
            T = np.hstack((T0, T1, T2, T3, T4))
            theta_hat = pinv(T.T @ np.diag(w) @ T) @ T.T @ np.diag(w) @ vecG

            b0_hat = theta_hat[:n_bar]
            B0_hat = utils.half_vectorize_inverse(b0_hat)
            err_b0_hat[:, nn, kk] = np.squeeze(b0 - b0_hat)

            b1_hat = theta_hat[n_bar:2 * n_bar]
            B1_hat = utils.half_vectorize_inverse(b1_hat)
            err_b1_hat[:, nn, kk] = np.squeeze(b1 - b1_hat)

            b2_hat = theta_hat[2 * n_bar: 3 * n_bar]
            B2_hat = utils.half_vectorize_inverse(b2_hat)
            err_b2_hat[:, nn, kk] = np.squeeze(b2 - b2_hat)

            b3_hat = theta_hat[3 * n_bar: 4 * n_bar]
            B3_hat = utils.half_vectorize_inverse(b3_hat)
            err_b3_hat[:, nn, kk] = np.squeeze(b3 - b3_hat)

            b4_hat = theta_hat[-n_bar:]
            B4_hat = utils.half_vectorize_inverse(b4_hat)
            err_b4_hat[:, nn, kk] = np.squeeze(b4 - b4_hat)

            # relative position - centered
            Y0_bar_hat[:, :, nn, kk] = utils.cMDS(B0_hat, center=False)[:nDim, :]
            err_y0_hat[:, nn, kk], H_Y0 = procrustes_error(Y0_bar_hat[:, :, nn, kk], Y0_bar)

            # relative acceleration - centered
            Y2_tilde_hat[:, :, nn, kk] = utils.cMDS(4 * B4_hat, center=False)[:nDim, :]
            err_y2_hat[:, nn, kk], H_Y2 = procrustes_error(Y2_tilde_hat[:, :, nn, kk], Y2_bar)

            Y1_bar_hat[:, :, nn, kk], H2_hat[:, :, nn, kk] = utils.solve_lyapunov_like_eqns(Y0_bar_hat[:, :, nn, kk], B1_hat, Y2_tilde_hat[:, :, nn, kk], 2 * B3_hat, M=nDim, N=N)
            Y2_bar_hat = H2_hat[:, :, nn, kk].T @ Y2_tilde_hat[:, :, nn, kk]

            # relative velocity error
            err_y1_hat[:, nn, kk], H_Y1 = procrustes_error(Y1_bar_hat[:, :, nn, kk], Y1_bar)

            # With ACCELEROMETER
            S = np.zeros((nDim, N, len(t_acc)))
            vecS = np.zeros((n_ACC * len(t_acc), 1))
            for jj in range(len(t_acc)):
                vec_eta_acc = noise_val_acc[jj * n_ACC: (jj + 1) * n_ACC]
                eta_acc = utils.vectorize_inverse(vec_eta_acc, rows=nDim, cols=N)
                S[:, :, jj] = H2 @ Y2_bar + eta_acc
                vecS[jj * n_ACC: (jj + 1) * n_ACC] = utils.vectorize(S[:, :, jj])

            # Forming matrix V
            I_nel = np.identity(n_ACC)
            V0 = np.kron(np.ones(t_acc.shape), I_nel)
            V1 = np.kron(t_acc, I_nel)
            V2 = np.kron(np.square(t_acc), I_nel)
            V = np.hstack((V0, V1, V2))

            # Solving for parameter thetaFalse
            alpha_hat = pinv(V.T @ V) @ V.T @ vecS
            # alpha_hat_ls = np.linalg.lstsq(V, tau)

            y2_tilde_acc = alpha_hat[:n_ACC]
            Y2_tilde_acc[:, :, nn, kk] = utils.vectorize_inverse(y2_tilde_acc, rows=nDim, cols=N)
            err_y2_acc[:, nn, kk], H_Y2 = procrustes_error(Y2_tilde_acc[:, :, nn, kk], Y2_bar)

            # Forming matrix T
            T_acc = np.hstack((T0, T1, T2, T3))
            y2ty2 = utils.half_vectorize(Y2_tilde_acc[:, :, nn, kk].T @ Y2_tilde_acc[:, :, nn, kk])
            vecG_acc = vecG - 0.25 * T4.dot(y2ty2)

            # Solving for parameter theta
            theta_hat_acc = pinv(T_acc.T @ T_acc) @ T_acc.T @ vecG_acc
            # theta_hat_ls = np.linalg.lstsq(T, g)

            b0_acc = theta_hat_acc[:n_bar]
            B0_acc = utils.half_vectorize_inverse(b0_acc)
            err_b0_acc[:, nn, kk] = np.squeeze(b0 - b0_acc)

            b1_acc = theta_hat_acc[n_bar:2 * n_bar]
            B1_acc = utils.half_vectorize_inverse(b1_acc)
            err_b1_acc[:, nn, kk] = np.squeeze(b1 - b1_acc)

            b2_acc = theta_hat_acc[2 * n_bar: 3 * n_bar]
            B2_acc = utils.half_vectorize_inverse(b2_acc)
            err_b2_acc[:, nn, kk] = np.squeeze(b2 - b2_acc)

            b3_acc = theta_hat_acc[3 * n_bar: 4 * n_bar]
            B3_acc = utils.half_vectorize_inverse(b3_acc)
            err_b3_acc[:, nn, kk] = np.squeeze(b3 - b3_acc)

            # relative position - centered
            Y0_bar_acc[:, :, nn, kk] = utils.cMDS(B0_acc, center=False)[:nDim, :]
            err_y0_acc[:, nn, kk], H_Y0 = procrustes_error(Y0_bar_acc[:, :, nn, kk], Y0_bar)

            # relative velocity error
            Y1_bar_acc[:, :, nn, kk], H2_acc[:, :, nn, kk] = utils.solve_lyapunov_like_eqns(Y0_bar_acc[:, :, nn, kk], B1_acc, Y2_tilde_acc[:, :, nn, kk], 2 * B3_acc, M=nDim, N=N)
            Y2_bar_acc = H2_acc[:, :, nn, kk] .T @ Y2_tilde_acc[:, :, nn, kk]
            err_y1_acc[:, nn, kk], H_Y1 = procrustes_error(Y1_bar_acc[:, :, nn, kk], Y1_bar)

        rmse_b0_hat[kk] = np.sqrt(np.sum(np.square(norm(err_b0_hat[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b1_hat[kk] = np.sqrt(np.sum(np.square(norm(err_b1_hat[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b2_hat[kk] = np.sqrt(np.sum(np.square(norm(err_b2_hat[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b3_hat[kk] = np.sqrt(np.sum(np.square(norm(err_b3_hat[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b4_hat[kk] = np.sqrt(np.sum(np.square(norm(err_b4_hat[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_y0_hat[kk] = np.sqrt(np.sum(np.square(norm(err_y0_hat[:, :, kk], axis=0))) / N_EXP) / (nDim * N)
        rmse_y1_hat[kk] = np.sqrt(np.sum(np.square(norm(err_y1_hat[:, :, kk], axis=0))) / N_EXP) / (nDim * N)
        rmse_y2_hat[kk] = np.sqrt(np.sum(np.square(norm(err_y2_hat[:, :, kk], axis=0))) / N_EXP) / (nDim * N)

        rmse_b0_acc[kk] = np.sqrt(np.sum(np.square(norm(err_b0_acc[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b1_acc[kk] = np.sqrt(np.sum(np.square(norm(err_b1_acc[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b2_acc[kk] = np.sqrt(np.sum(np.square(norm(err_b2_acc[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_b3_acc[kk] = np.sqrt(np.sum(np.square(norm(err_b3_acc[:, :, kk], axis=0))) / N_EXP) / n_bar
        rmse_y0_acc[kk] = np.sqrt(np.sum(np.square(norm(err_y0_acc[:, :, kk], axis=0))) / N_EXP) / (nDim * N)
        rmse_y1_acc[kk] = np.sqrt(np.sum(np.square(norm(err_y1_acc[:, :, kk], axis=0))) / N_EXP) / (nDim * N)
        rmse_y2_acc[kk] = np.sqrt(np.sum(np.square(norm(err_y2_acc[:, :, kk], axis=0))) / N_EXP) / (nDim * N)

    if save_flag:
            out_name = 'mc_2step_acc_snr_ctr'
            output = pathlib.Path(out_name)
            results_path = output.with_suffix('.npz')
            with open(results_path, 'xb') as results_file:
                np.savez(results_file, snr=SNR_gains, rmse_y0_hat=rmse_y0_hat, rmse_y1_hat=rmse_y1_hat,
                         rmse_y2_hat=rmse_y2_hat, rmse_b0_hat=rmse_b0_hat, rmse_b1_hat=rmse_b1_hat,
                         rmse_b2_hat=rmse_b2_hat, rmse_b3_hat=rmse_b3_hat, rmse_b4_hat=rmse_b4_hat,
                         rmse_b0_acc=rmse_b0_acc, rmse_b1_acc=rmse_b1_acc, rmse_b2_acc=rmse_b2_acc,
                         rmse_b3_acc=rmse_b3_acc, rmse_y0_acc=rmse_y0_acc, rmse_y1_acc=rmse_y1_acc,
                         rmse_y2_acc=rmse_y2_acc, Y0_bar_hat=Y0_bar_hat, Y1_bar_hat=Y1_bar_hat,
                         Y2_tilde_hat=Y2_tilde_hat, Y0_bar_acc=Y0_bar_acc, Y1_bar_acc=Y1_bar_acc,
                         Y2_tilde_acc=Y2_tilde_acc, err_b0_hat=err_b0_hat, err_b1_hat=err_b1_hat, err_b2_hat=err_b2_hat,
                         err_b3_hat=err_b3_hat, err_b4_hat=err_b4_hat, err_y0_hat=err_y0_hat, err_y1_hat=err_y1_hat,
                         err_y2_hat=err_y2_hat, err_b0_acc=err_b0_acc, err_b1_acc=err_b1_acc, err_b2_acc=err_b2_acc,
                         err_b3_acc=err_b3_acc, err_y0_acc=err_y0_acc, err_y1_acc=err_y1_acc, err_y2_acc=err_y2_acc,
                         H2_hat=H2_hat, H2_acc=H2_acc)

    print('finished!')