import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import utils_rl as utils
import math
import matplotlib as mpl
mpl.use('Qt5Agg')

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

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
    STD_DIST = 0.1
    OPTIONS = np.array([1, 1]) # first value for crlb main and second for crlb gtwr

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
    one = np.ones((N, 1))
    C = np.eye(N) - (one @ one.T) / N
    Y0_bar = Y0 @ C
    Y1_bar = Y1 @ C
    B0 = Y0_bar.T @ Y0_bar
    b0 = utils.half_vectorize(B0)
    B1 = Y0_bar.T @ Y1_bar + Y1_bar.T @ Y0_bar
    b1 = utils.half_vectorize(B1)
    B2 = Y1_bar.T @ Y1_bar
    b2 = utils.half_vectorize(B2)

    # jacobians for relative position and velocity
    if OPTIONS[0]: # MAIN 2 STEP
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

        J1_main = np.zeros((n_bar, N * nDim))
        for ii in range(n_bar):  # # going through all possible pairwise distance indices
            for jj in range(N):  # for each node position
                if idx[0][ii] == idx[1][ii] and jj == idx[0][ii]:
                    J1_main[ii, jj * nDim: (jj + 1) * nDim] = 2. * (Y1_bar[:, idx[0][ii]])
                elif jj == idx[0][ii]:  # jj == idx[0][ii]:
                    J1_main[ii, jj * nDim: (jj + 1) * nDim] = Y1_bar[:, idx[1][ii]]
                elif jj == idx[1][ii]:  # jj == idx[1][ii]:
                    J1_main[ii, jj * nDim: (jj + 1) * nDim] = Y1_bar[:, idx[0][ii]]
                else:
                    J1_main[ii, jj * nDim: (jj + 1) * nDim] = np.zeros(2)

        Dm = utils.duplication_matrix_char(N)
        M = -0.5 * pinv(Dm) @ np.kron(C.T, C) @ Dm

        rmse_main_y0 = np.zeros(len(K_array))
        fim_main_y0 = np.zeros((nDim * N, nDim * N, len(K_array)))
        rmse_main_y1 = np.zeros(len(K_array))
        fim_main_y1 = np.zeros((nDim * N, nDim * N, len(K_array)))
        rmse_main_b0 = np.zeros(len(K_array))
        rmse_main_b1 = np.zeros(len(K_array))
        rmse_main_b2 = np.zeros(len(K_array))

    if OPTIONS[1]: # GTWR
        d0 = np.sqrt(utils.half_vectorize(utils.edm(Y0), skew=True))
        idx = np.triu_indices(N, 1)
        J0_gtwr = np.zeros((N_bar, N * nDim))
        for ii in range(N_bar):  # # going through all possible pairwise distance indices
            for jj in range(N):  # for each node position
                if jj == idx[0][ii]:
                    J0_gtwr[ii, jj * nDim: (jj + 1) * nDim] = (Y0[:, idx[0][ii]] - Y0[:, idx[1][ii]]) / d0[ii]
                elif jj == idx[1][ii]:
                    J0_gtwr[ii, jj * nDim: (jj + 1) * nDim] = - (Y0[:, idx[0][ii]] - Y0[:, idx[1][ii]]) / d0[ii]
                else:
                    J0_gtwr[ii, jj * nDim: (jj + 1) * nDim] = np.zeros(2)

        # jacobian for relative velocity
        J1_gtwr = np.zeros((N_bar, N * nDim))
        for ii in range(N_bar):  # # going through all possible pairwise distance indices
            for jj in range(N):  # for each node position
                if jj == idx[0][ii]:
                    J1_gtwr[ii, jj * nDim: (jj + 1) * nDim] = 2 * (Y1[:, idx[0][ii]] - Y1[:, idx[1][ii]])
                elif jj == idx[1][ii]:
                    J1_gtwr[ii, jj * nDim: (jj + 1) * nDim] = - 2 * (Y1[:, idx[0][ii]] - Y1[:, idx[1][ii]])
                else:
                    J1_gtwr[ii, jj * nDim: (jj + 1) * nDim] = np.zeros(2)

        # to account for the scaling
        L = 4
        gamma = np.zeros(L + 1)
        for ii in range(L + 1):
            # to account for the factorials
            # gamma[ii] = c * math.factorial(ii)
            gamma[ii] = math.factorial(ii)
        scale_mat = np.kron(np.diag(gamma), np.identity(N_bar))

        # RANGE DERIVATIVE - GROUND TRUTH
        r_derivatives = np.zeros((N_bar, 3))
        for ii in range(N_bar):
            x1 = Y0[:, idx[0][ii]]
            x2 = Y0[:, idx[1][ii]]
            v1 = Y1[:, idx[0][ii]]
            v2 = Y1[:, idx[1][ii]]
            a1 = np.zeros(2)
            a2 = np.zeros(2)
            r_derivatives[ii, :] = utils.distance_derivatives_time(x1, v1, a1, x2, v2, a2)
            # tmp = utils.range_taylor_coeffs(x1, v1, a1, x2, v2, a2)
            # print(tmp - r_derivatives[ii, :])

        R = np.diag(r_derivatives[:, 0])
        dotR = np.diag(r_derivatives[:, 1])
        ddotR = np.diag(r_derivatives[:, 2])

        rmse_gtwr_y0 = np.zeros(len(K_array))
        fim_gtwr_y0 = np.zeros((nDim * N, nDim * N, len(K_array)))
        rmse_gtwr_y1 = np.zeros(len(K_array))
        fim_gtwr_y1 = np.zeros((nDim * N, nDim * N, len(K_array)))
        rmse_gtwr_r = np.zeros(len(K_array))
        rmse_gtwr_dotr = np.zeros(len(K_array))
        rmse_gtwr_ddotr = np.zeros(len(K_array))


    for mm in range(len(K_array)):
        K = K_array[mm]
        print([mm, K])
        t_pwd = np.linspace(-tend, tend, K + 1).reshape(K + 1, 1)

        if OPTIONS[0]: # MAIN 2 STEP
            # Forming matrix T
            I_nbar = np.identity(n_bar)
            T0 = np.kron(np.ones(t_pwd.shape), I_nbar)
            T1 = np.kron(t_pwd, I_nbar)
            T2 = np.kron(np.square(t_pwd), I_nbar)
            T = np.hstack((T0, T1, T2))

            # entries corresponding to diagonal entries of the matrix form must be zero.
            # But here it is fine because corresponding entries are 0 in diag(bold_d[:, ii])
            # Sigma_d = STD_DIST ** 2 * np.identity(N_bar)
            Sigma_d = STD_DIST ** 2 * np.identity(n_bar)

            # Lm = utils.elim_mat(N)
            # Dm = utils.duplication_matrix_char(N)
            # tmp = utils.off_diag_select_matrix(N)
            # Sm = utils.selection_matrix(N)
            # M = -0.5 * Lm @ np.kron(C.T, C) @ pinv(Sm)
            # M = -0.5 * pinv(Dm) @ np.kron(C.T, C) @ Dm
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

            Sigma_b0 = Sigma_theta[:n_bar, :n_bar]
            Sigma_b1 = Sigma_theta[n_bar:2 * n_bar, n_bar:2 * n_bar]
            Sigma_b2 = Sigma_theta[2 * n_bar:, 2 * n_bar:]
            rmse_main_b0[mm] = np.sqrt(np.trace(Sigma_b0)) / n_bar
            rmse_main_b1[mm] = np.sqrt(np.trace(Sigma_b1)) / n_bar
            rmse_main_b2[mm] = np.sqrt(np.trace(Sigma_b2)) / n_bar

            # CRLB for relative position estimate
            fim_main_y0[:, :, mm] = J0_main.T @ pinv(Sigma_b0) @ J0_main
            crlb_main_y0 = pinv(fim_main_y0[:, :, mm])
            rmse_main_y0[mm] = np.sqrt(np.trace(crlb_main_y0)) / (N * nDim)

            # CRLB for relative velocity estimate
            fim_main_y1[:, :, mm] = J1_main.T @ pinv(Sigma_b2) @ J1_main
            crlb_main_y1 = pinv(fim_main_y1[:, :, mm])
            rmse_main_y1[mm] = np.sqrt(np.trace(crlb_main_y1)) / (N * nDim)


        if OPTIONS[1]: # GTWR
            # Forming matrix
            I_nel = np.identity(N_bar)
            V0 = np.kron(np.ones(t_pwd.shape), I_nel)
            V1 = np.kron(t_pwd, I_nel)
            V2 = np.kron(np.square(t_pwd), I_nel)
            V3 = np.kron(np.power(t_pwd, 3), I_nel)
            V4 = np.kron(np.power(t_pwd, 4), I_nel)
            # V = np.hstack((V0, V1, V2, V3))
            V_gtwr = np.hstack((V0, V1, V2, V3, V4))

            # covariance of the measurements tau is k^2 Sigma with k = 1/c
            # I have not scaled with k^2 to keep the computation accurate in terms of precision
            # later I can scale the result accordingly to account for this
            # Sigma = (cst.STD_DIST ** 2 / (c ** 2)) * np.identity(N_bar * K)
            Sigma = STD_DIST ** 2 * np.identity(N_bar * (K + 1))

            fim_der = V_gtwr.T @ pinv(Sigma) @ V_gtwr
            Sigma_theta_scaled = pinv(fim_der)

            # plotting the covariance matrix to inspect structure
            # plt.imshow(Sigma_theta)
            # plt.colorbar()
            # plt.show()

            # THIS IS WHERE I CORRECT FOR THE SCALING:
            #   Earlier I should have divided by c^2 and here I should have multiplied with c^2
            #   So in all I should get the correct result here
            Sigma_theta = scale_mat.T @ Sigma_theta_scaled @ scale_mat

            # assigning covariance to individual scaled derivatives
            Sigma_der = []
            for ii in range(L):
                Sigma_der.append(Sigma_theta[ii * N_bar : (ii + 1) * N_bar, ii * N_bar : (ii + 1) * N_bar])

            # RMSE for range derivatives
            rmse_gtwr_r[mm] = np.sqrt(np.trace(Sigma_der[0])) / Sigma_der[0].shape[0]
            rmse_gtwr_dotr[mm] = np.sqrt(np.trace(Sigma_der[1])) / Sigma_der[1].shape[0]
            rmse_gtwr_ddotr[mm] = np.sqrt(np.trace(Sigma_der[2])) / Sigma_der[2].shape[0]

            #Cramer-Rao Lower Bound - Relative Position
            fim_gtwr_y0[:, :, mm] = J0_gtwr.T @ pinv(Sigma_der[0]) @ J0_gtwr
            crlb_gtwr_y0 = pinv(fim_gtwr_y0[:, :, mm])
            rmse_gtwr_y0[mm] = np.sqrt(np.trace(crlb_gtwr_y0)) / (N * nDim)

            #Cramer-Rao Lower Bound - Relative Velocity
            # covariance matrix for "velocity measurements"
            Sigma_bar_y1 = R.T @ Sigma_der[2] @ R + ddotR.T @ Sigma_der[0] @ ddotR + 4 * dotR.T @ Sigma_der[1] @ dotR
            Sigma_y1 = Sigma_bar_y1

            fim_gtwr_y1[:, :, mm] = J1_gtwr.T @ pinv(Sigma_y1) @ J1_gtwr
            crlb_gtwr_y1 = pinv(fim_gtwr_y1[:, :, mm])
            rmse_gtwr_y1[mm] = np.sqrt(np.trace(crlb_gtwr_y1)) / (N * nDim)
            print([rmse_gtwr_y0[mm], rmse_gtwr_y1[mm]])

    '''
        plots
    '''
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_yscale('log')
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_yscale('log')

    # # range derivatives
    # ax0.plot(K_array, rmse_r, 'bo--', label='range')
    # ax0.plot(K_array, rmse_dotr, 'ro--', label='range_rate')
    # ax0.plot(K_array, rmse_ddotr, 'ko--', label='range_rate_rate')

    # relative position
    ax0.plot(K_array, rmse_main_y0, 'bo-', label='proposed')
    ax0.plot(K_array, rmse_gtwr_y0, 'ko--', label='Rajan et al.')

    # relative velocity
    ax1.plot(K_array, rmse_main_y1, 'bo-', label='proposed')
    ax1.plot(K_array, rmse_gtwr_y1, 'ko--', label='Rajan et al.')

    ax0.grid(True)
    ax0.legend()
    ax1.grid(True)
    ax1.legend()
    plt.show()

    if save_flag:
        out_name = 'crlb_vel2'
        output = pathlib.Path(out_name)
        results_path = output.with_suffix('.npz')
        with open(results_path, 'xb') as results_file:
            if OPTIONS[0] and not OPTIONS[1]:
                np.savez(results_file, N=N, K=K_array, rmse_main_y0=rmse_main_y0, rmse_main_y1=rmse_main_y1,
                         rmse_main_b0=rmse_main_b0, rmse_main_b1=rmse_main_b1, rmse_main_b2=rmse_main_b2)
            elif not OPTIONS[0] and OPTIONS[1]:
                np.savez(results_file, N=N, K=K_array, rmse_gtwr_r=rmse_gtwr_r, rmse_gtwr_dotr=rmse_gtwr_dotr,
                         rmse_gtwr_ddotr=rmse_gtwr_ddotr, rmse_gtwr_y0=rmse_gtwr_y0, rmse_gtwr_y1=rmse_gtwr_y1)
            else:
                np.savez(results_file, N=N, K=K_array, rmse_main_y0=rmse_main_y0, rmse_main_y1=rmse_main_y1,
                         rmse_main_b0=rmse_main_b0, rmse_main_b1=rmse_main_b1, rmse_main_b2=rmse_main_b2,
                         rmse_gtwr_r=rmse_gtwr_r, rmse_gtwr_dotr=rmse_gtwr_dotr, rmse_gtwr_ddotr=rmse_gtwr_ddotr,
                         rmse_gtwr_y0=rmse_gtwr_y0, rmse_gtwr_y1=rmse_gtwr_y1)

    print('finished!')