import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp
from scipy.sparse import csr_matrix
import sympy as sym
from scipy.optimize import minimize, NonlinearConstraint, lsq_linear
from numpy.linalg import pinv, lstsq, norm, svd
from scipy.linalg import orthogonal_procrustes

def edm(X):
    d, n = X.shape

    one = np.ones((n, 1))
    G = X.transpose().dot(X)
    g = G.diagonal().reshape((n, 1))
    D = g.dot(one.transpose()) + one.dot(g.transpose()) - 2.0 * G

    return D

def pairwise_distance(X):
    d, n = X.shape
    pair_dist = np.zeros((n, n))

    for ii in range(n):
        for jj in range(n):
            pair_dist[ii, jj] = np.linalg.norm( X[:, ii] - X[:, jj] )

    return pair_dist

def double_center(D, opt=True):
    _, n = D.shape
    one = np.ones((n, 1))

    if opt:
        # move centroid to origin
        J = np.identity(n) - one.dot(one.transpose()) / n
    else:
        # move first point to origin
        s = np.zeros((n, 1))
        s[0] = 1.
        J = np.identity(n) - s.dot(one.transpose())

    return -0.5 * J.transpose().dot(D).dot(J)

def cMDS(D, center=True):
    if center:
        G = double_center(D)
    else:
        G = D

    eig_val, eig_vec = np.linalg.eigh(G)
    indices = np.flipud(sorted(range(len(eig_val)), key=lambda k: eig_val[k]))

    # sorting eigenvalues and corresponding eigenvectors in descending order
    sorted_eig_val = eig_val[indices]
    # fast sorting of numpy arrays (https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array)
    idx = np.empty_like(indices)
    idx[indices] = np.arange(len(indices))
    sorted_eig_vec = eig_vec[:, idx]

    return np.diag(np.sqrt(sorted_eig_val)).dot(sorted_eig_vec.transpose())

def gt(X_list, n):
    x = np.zeros((len(X_list), n))
    y = np.zeros((len(X_list), n))
    xrel = np.zeros((len(X_list), n))
    yrel = np.zeros((len(X_list), n))

    delta_s = np.zeros((len(X_list) , n))
    rots = np.zeros((len(X_list)))
    # x_delta_s = np.zeros((len(X_list), n))
    inc_rots = np.zeros((len(X_list)))

    for ii in range(len(X_list)):
        # absolute coordinates
        x[ii, :] = X_list[ii][0, :]
        y[ii, :] = X_list[ii][1, :]

        # relative coordinates
        xrel[ii, :] = x[ii, :] - x[ii, 0]
        yrel[ii, :] = y[ii, :] - y[ii, 0]

        # orientation of line joining node k and k'
        rots[ii] = np.arctan2(yrel[ii, 1], xrel[ii, 1])

        if ii:
            # NOTE: np.sqrt and np.square give element-wise operations
            # relative distance travelled
            # x_delta_s[ii, :] = np.sqrt(np.square(x[ii, :] - x[ii - 1, :]) + np.square(y[ii, :] - y[ii - 1, :]))
            delta_s[ii, :] = np.sqrt(np.square(xrel[ii, :] - xrel[ii - 1, :]) + np.square(yrel[ii, :] - yrel[ii - 1, :]))
    # incremental angle between different time frames for nodes k and k'
    inc_rots[1:] = np.diff(rots)

    return x, y, xrel, yrel, delta_s, inc_rots


def mds_consistency_check(X, Xprev, plot_flag):
    '''
    :param X: current solution provided by MDS
    :param Xprev: previous solution provided by MDS
    :return: Xcons: consistent MDS solution
    '''
    d, n = X.shape
    Xcons = X
    Xdiff = X - Xprev

    if plot_flag:
        plt.figure()
        plt.plot(X[0, :], X[1, :], 'bo')
        plt.plot(Xprev[0, :], Xprev[1, :], 'ko')

        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return Xcons

def half_vectorize(X, skew=False, ch=False):
    n1, n2 = X.shape
    if not ch:
        assert np.any(X - X.transpose() < 1e-6)

    # get the lower triangle indices to form a vector
    if skew:
        # idx = np.tril_indices(n1, -1)
        idx = np.triu_indices(n1, 1)
    else:
        # idx = np.tril_indices(n1)
        idx = np.triu_indices(n1)
    n_el = len(idx[0])

    if ch:
        vech = np.chararray((n_el, 1))
    else:
        vech = np.zeros((n_el, 1))

    for ii in range(n_el):
        vech[ii, 0] = X[idx[0][ii], idx[1][ii]]

    return vech

def half_vectorize_inverse(v, skew=False):
    n_el = len(v)

    if skew:
        n = int((1 + np.sqrt(1 + 4 * (2 * n_el))) / 2)
    else:
        n = int((-1 + np.sqrt(1 + 4 * (2 * n_el))) / 2)

    X = np.zeros((n, n))

    if skew:
        # idx = np.tril_indices(n, -1)
        idx = np.triu_indices(n, 1)
        for ii in range(n_el):
            X[idx[0][ii], idx[1][ii]] = v[ii]
        X += X.transpose()
    else:
        # idx = np.tril_indices(n)
        idx = np.triu_indices(n)
        for ii in range(n_el):
            X[idx[0][ii], idx[1][ii]] = v[ii]
            # using the fact that the opposite of the indexing gives the upper half
            X[idx[1][ii], idx[0][ii]] = v[ii]

    return X

def vectorize(A, ch=False):
    rows, cols = A.shape
    if ch:
        vecA = np.chararray((rows * cols, 1))
    else:
        vecA = np.zeros((rows * cols, 1))

    for ii in range(cols):
        # vecA[ii * cols: (ii + 1) * cols] = A[:, ii].reshape((rows, 1))
        vecA[ii * rows: (ii + 1) * rows] = A[:, ii].reshape((rows, 1))

    return vecA

def vectorize_inverse(v, rows=None, cols=None):
    n = len(v)

    if not rows or not cols:
        cols = int(np.sqrt(n))
        rows = int(np.sqrt(n))

    A = np.zeros((rows, cols))
    for ii in range(cols):
        A[:, ii] = v[ii * rows: (ii + 1) * rows].reshape(rows)

    return A

# source: https://stackoverflow.com/questions/60678746/compute-commutation-matrix-in-numpy-scipy-efficiently
def commutation_matrix(m, n):
    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K

# source: https://en.wikipedia.org/wiki/Commutation_matrix
def commutation_matrix_wiki(m, n):
    K = np.zeros((n * m, m* n))
    for ii in range(m):
        for jj in range(n):
            K[ii + m * jj, jj + n * ii] = 1

    return K

def sdp_edm(D, nDim, n):
    Dtilde = double_center(D)

    # Setting up the cvx problem
    Y = cp.Variable((n, n), symmetric=True)
    Z = cp.Variable((nDim, n))

    # Schur complement
    I = np.identity(nDim)
    In = np.identity(n)
    M = cp.bmat([[Y, Z.T], [Z, I]])

    # for indexing in symmetric matrix
    idx = np.tril_indices(n, -1)
    n_el = len(idx[0])

    constraints = [M >> 0]
    constraints += [
        (In[:, idx[0][ii]] - In[:, idx[1][ii]]) @ Y @ (In[:, idx[0][ii]] - In[:, idx[1][ii]]).T == D[
            idx[0][ii], idx[1][ii]] for ii in range(n_el)
    ]

    # adding the diagonal element constraints breaks the algorithm
    # constraints += [
    #     Y[ii, ii] == D[ii, ii] for ii in range(n)
    # ]

    prob = cp.Problem(cp.Minimize(cp.trace(Y - Dtilde)), constraints)
    prob.solve()

    return prob, Y.value

def reflect_along_axis_2D(vel_vec, axis_vec):
    #source: https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    # in the source, for our case, n should be the line not the normal to the line
    t_vec = vel_vec - 2 * np.dot(vel_vec, axis_vec) * axis_vec

    # now the vector has the opposite sign, so it has to be corrected
    return -t_vec

def distance_derivatives_time(x1, v1, a1, x2, v2, a2):
    delx = x2 - x1
    delv = v2 - v1
    dela = a2 - a1
    d = np.sqrt(delx.transpose().dot(delx))

    # first distance derivative
    first_der = delx.transpose().dot(delv) / d

    # second distance derivative
    first_term = -np.square(delx.transpose().dot(delv)) / d ** 3
    second_term = (delv.transpose().dot(delv) + delx.transpose().dot(dela)) / d
    second_der = first_term + second_term

    return np.array([d, first_der, second_der])

def range_taylor_coeffs(x1, v1, a1, x2, v2, a2, t=0.):
    dely0 = x2 - x1
    dely1 = v2 - v1
    dely2 = a2 - a1

    # zeroth order derivative at time t for L = 2
    delx = dely0 + dely1 * t + 0.5 * dely2 * t**2
    delv = dely1 + dely2 * t
    dela = dely2

    # zeroth order derivative
    d = np.sqrt(delx.transpose().dot(delx))

    # first distance derivative
    first_der = delx.transpose().dot(delv) / d

    # second distance derivative
    first_term = -np.square(delx.transpose().dot(delv)) / d ** 3
    second_term = (delv.transpose().dot(delv) + delx.transpose().dot(dela)) / d
    second_der = first_term + second_term

    return np.array([d, first_der, second_der])

def duplication_matrix(N):
    # source: https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices
    idx = np.tril_indices(N)
    N_bar = int(N * (N + 1) / 2.)
    # duplication matrix initialization
    D = np.zeros((N ** 2, N_bar))

    for ii in range(N):#range(N ** 2):
        for jj in range(N):#range(int(N * (N + 1) / 2.)):
            if jj <= ii:
                u_ij = np.zeros((N_bar, 1))
                # tmp_idx = int((jj - 1) * N + ii - (jj * (jj - 1)) / 2.)
                tmp_idx = int(jj * N + ii + 1. - ((jj + 1.) * jj) / 2.) - 1
                # if tmp_idx < N_bar and tmp_idx > - N_bar:
                #     u_ij[tmp_idx] = 1.
                u_ij[tmp_idx] = 1.
                T_ij = np.zeros((N, N))
                T_ij[ii, jj] = 1.
                T_ij[jj, ii] = 1.

                tmp = u_ij.dot(vectorize(T_ij).reshape((N ** 2, 1)).transpose())
                D += tmp.transpose()

    return D

def duplication_matrix_new(N):
    # source: https://math.stackexchange.com/questions/3984389/on-the-relation-between-the-vectorization-and-the-half-vectorization
    N_bar = int(N * (N + 1) / 2.)

    # duplication matrix initialization
    D = np.zeros((N ** 2, N_bar))

    for ii in range(N):
        for jj in range(N):
            for gg in range(N):
                for hh in range(N):
                    if gg <= hh:
                        rIdx = (ii + 1) * (jj + 1) - 1
                        cIdx = (gg + 1) * (hh + 1) - 1
                        if rIdx < N ** 2 and cIdx < N_bar:
                            D[rIdx, cIdx] = 1. / 2. * (int(ii == gg) * int(jj == hh) + int(ii == hh) * int(jj == gg))
    return D

def duplication_matrix_char(N):
    N_bar = int(N * (N + 1) / 2.)

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '+', '-', '=']
    idx = np.triu_indices(N)

    # duplication matrix initialization
    D = np.zeros((N ** 2, N_bar))
    D_char = np.chararray((N, N))

    for ii in range(N_bar):
        D_char[idx[0][ii], idx[1][ii]] = letters[ii]
        D_char[idx[1][ii], idx[0][ii]] = letters[ii]

    # vectorizing D_char
    vecD_char = vectorize(D_char, ch=True)

    # half vectorize D_char
    vechD_char = half_vectorize(D_char, ch=True)

    for jj in range(N ** 2):
        idx = np.where(vechD_char == vecD_char[jj])
        D[jj, idx[0]] = 1.

    return D

def selection_matrix(N):
    # number of unique elements in a symmetric matrix with zero diagonal
    N_bar = int(N * (N - 1) / 2.)

    # number of unique elements in a symmetric matrix
    n_bar = int(N * (N + 1) / 2.)

    L = elim_mat(N)

    # positions where the zero rows are to be added
    S = np.zeros((N_bar, N ** 2))
    zero_row_ids = np.concatenate((np.array([0]), np.cumsum(np.arange(N, 1, -1))))
    counter = 0
    for ii in range(N_bar):
        if ii in zero_row_ids:
            continue
        else:
            S[counter, :] = L[ii, :]
            counter += 1

    return S

def off_diag_select_matrix(N):
    # number of unique elements in a symmetric matrix with zero diagonal
    N_bar = int(N * (N - 1) / 2.)

    # number of unique elements in a symmetric matrix
    n_bar = int(N * (N + 1) / 2.)

    I = np.identity(N_bar)

    # positions where the zero rows are to be added
    S = np.zeros((n_bar, N_bar))
    zero_row_ids = np.cumsum(np.arange(N, 1, -1))
    counter = 0
    for ii in range(n_bar - 1):
        if ii + 1 in zero_row_ids:
            continue
        else:
            S[ii + 1, :] = I[counter, :]
            counter += 1

    return S


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

def solve_lyapunov_like_eqns_sym(A, B, C, D, M=2, N=4):
    '''
    Solve Lyapunov-like equations of the form
    :param B: square matrix corresponding to position and velocity
    :param D: square matrix corresponding to acceleration and velocity
    :return: relative velocity and rotation
    '''

    # SVD
    UA, lmdA, VTA = np.linalg.svd(A, full_matrices=True)
    VA = VTA.transpose()
    UC, lmdC, VTC = np.linalg.svd(C, full_matrices=True)
    VC = VTC.transpose()

    B_tilde = VTA.dot(B).dot(VA)
    D_tilde = VTC.dot(D).dot(VC)

    B11_tilde = B_tilde[:M, :M]
    B12_tilde = B_tilde[:M, M:]
    B22_tilde = B_tilde[M:, M:]
    D11_tilde = D_tilde[:M, :M]
    D12_tilde = D_tilde[:M, M:]
    D22_tilde = D_tilde[M:, M:]

    Y2_est = np.linalg.pinv(np.diag(lmdA)).dot(B12_tilde)
    Y2_bar_est = np.linalg.pinv(np.diag(lmdC)).dot(D12_tilde)
    y11_est = B11_tilde[0, 0] / (2. * lmdA[0])
    y22_est = B11_tilde[1, 1] / (2. * lmdA[1])
    y11_bar_est = D11_tilde[0, 0] / (2. * lmdC[0])
    y22_bar_est = D11_tilde[1, 1] / (2. * lmdC[1])

    K = np.kron(VTA, UA.transpose())
    Kinv = np.linalg.pinv(K)
    K_bar = np.kron(VTC, UC.transpose())
    idx = (N - M) * M
    K2 = K[-idx:]
    K2_bar = K_bar[-idx:]

    h1, h2, h3, h4, u, v = sym.symbols('h1 h2 h3 h4 u v')

    h = np.array([[h1, h2], [h3, h4]])
    y_new = np.vstack((np.array([[y11_est], [u], [v], [y22_est]]), vectorize(Y2_est)))
    I = np.identity(N)
    P_new = np.kron(I, h)
    y_bar_new = K_bar.dot(P_new).dot(Kinv).dot(y_new)

    unknowns = [h1, h2, h3, h4, h2 * u, h4 * u, h1 * v, h3 * v]
    n_meas = (N - M) * M + M
    Mat = np.zeros((n_meas, len(unknowns)))

    '''
        I want to remove the off-diagonal terms of a square matrix Y in the vectorized y
            - say Y is M * M matrix
            - location of M diagonal elements when you vectorize Y to y
                - k * (M + 1) for k = {0, 1, .., M}
                - in other form, running from 0 to M ** 2 with a gap of M + 1 
    '''
    idx_diag = np.arange(0, M ** 2, M + 1)
    counter = 0
    for ii in range(M * N):
        tmp = sym.Poly(y_bar_new[ii][0], h1, h2, h3, h4, u, v)
        if ii in idx_diag or ii > M ** 2 - 1:  # to allow for only diagonal indices in Y1
            # rearranging coefficients [h1, h2, h1*u, h1*v, h2*u, h2*v]
            for jj in range(len(unknowns)):
                Mat[counter, jj] = tmp.coeff_monomial(unknowns[jj])
            counter += 1

    # masked_array = np.ma.masked_where(Mat < 1e-6, Mat)
    # cmap = matplotlib.cm.spring  # Can be any colormap that you want after the cm
    # cmap.set_bad(color='black')
    # plt.figure()
    # plt.imshow(masked_array, cmap=cmap)
    # plt.show()
    # collecting known values of y_bar
    r = np.vstack((np.array([[y11_bar_est], [y22_bar_est]]), vectorize(Y2_bar_est)))
    unknowns = np.linalg.pinv(Mat.transpose().dot(Mat)).dot(Mat.transpose()).dot(r)

    # ub = np.array([1, 1, np.infty, np.infty, np.infty, np.infty])
    # lb = -1 * ub
    # res = lsq_linear(Mat, r.reshape(n_meas), bounds=(lb, ub))

    h1_val = unknowns[0][0]
    h2_val = unknowns[1][0]
    h3_val = unknowns[2][0]
    h4_val = unknowns[3][0]
    u_1 = unknowns[4][0] / h2_val
    u_2 = unknowns[5][0] / h4_val
    v_1 = unknowns[6][0] / h1_val
    v_2 = unknowns[7][0] / h3_val

    print("\n Rotation matrix")
    print(np.array([[h1_val, h2_val], [h3_val, h4_val]]))
    print("\n u[1]")
    print(np.array([u_1, u_2]))
    print("\n u[2]")
    print(np.array([v_1, v_2]))

    '''
        QCQP formulation
    '''
    #TODO

    '''
        scipy.optimize.minimize
    '''
    Y_est = np.hstack((np.array([[y11_est, v_1], [u_1, y22_est]]), Y2_est))
    Y_est2 = np.hstack((np.array([[y11_est, v_2], [u_2, y22_est]]), Y2_est))
    X_est = np.linalg.pinv(UA).dot(Y_est).dot(np.linalg.pinv(VA))
    X_est2 = np.linalg.pinv(UA).dot(Y_est2).dot(np.linalg.pinv(VA))
    H_est = np.array([[h1_val, h2_val], [h3_val, h4_val]])
    # H_est_sp = np.array([[h1_val_sp, -h2_val_sp], [h2_val_sp, h1_val_sp]])
    # H_est_min = np.array([[h1_val_min, -h2_val_min], [h2_val_min, h1_val_min]])

    return X_est, H_est

def solve_lyapunov_like_eqns(A, B, C, D, M=2, N=4, method='least_squares'):
    '''
    Solve Lyapunov-like equations of the form
    :param B: square matrix corresponding to position and velocity
    :param D: square matrix corresponding to acceleration and velocity
    :return: relative velocity and rotation
    '''

    # SVD
    UA, lmdA, VTA = np.linalg.svd(A, full_matrices=True)
    VA = VTA.transpose()
    UC, lmdC, VTC = np.linalg.svd(C, full_matrices=True)
    VC = VTC.transpose()

    B_tilde = VTA.dot(B).dot(VA)
    D_tilde = VTC.dot(D).dot(VC)

    B11_tilde = B_tilde[:M, :M]
    B12_tilde = B_tilde[:M, M:]
    B22_tilde = B_tilde[M:, M:]
    D11_tilde = D_tilde[:M, :M]
    D12_tilde = D_tilde[:M, M:]
    D22_tilde = D_tilde[M:, M:]

    Y2_est = np.linalg.pinv(np.diag(lmdA)).dot(B12_tilde)
    Y2_bar_est = np.linalg.pinv(np.diag(lmdC)).dot(D12_tilde)
    y11_est = B11_tilde[0, 0] / (2. * lmdA[0])
    y22_est = B11_tilde[1, 1] / (2. * lmdA[1])
    y11_bar_est = D11_tilde[0, 0] / (2. * lmdC[0])
    y22_bar_est = D11_tilde[1, 1] / (2. * lmdC[1])

    K = np.kron(VTA, UA.transpose())
    Kinv = np.linalg.pinv(K)
    K_bar = np.kron(VTC, UC.transpose())
    # idx = (N - M) * M

    Mat = np.zeros((M * N, 6))
    Matr = np.zeros((M * N, 6))
    Kinv_check = np.zeros(Kinv.shape)
    swap_idx = swap_rows(M, N)
    for ll in range(M * N):
        Kinv_check[ll, :] = Kinv[swap_idx[ll], :]

    T_h1 = np.zeros((M * N, M * N))
    T_h2 = np.zeros((M * N, M * N))
    Tr_h1 = np.zeros((M * N, M * N))
    Tr_h2 = np.zeros((M * N, M * N))
    y_tmp = np.vstack((np.array([[y11_est], [np.nan], [np.nan], [y22_est]]), vectorize(Y2_est)))
    # yr_tmp = np.vstack((np.array([[y11_est], [np.nan], [np.nan], [-y22_est]]), vectorize(np.array([[1, 0], [0, -1]]).dot(Y2_est))))
    for ii in range(M * N):
        for jj in range(M * N):
            for kk in range(M * N):
                # no reflection
                T_h1[ii, jj] += K_bar[ii, kk] * Kinv[kk, jj]
                T_h2[ii, jj] += np.power(-1, kk + 1) * K_bar[ii, kk] * Kinv_check[kk, jj]
                # with reflection
                Tr_h1[ii, jj] += np.power(-1, kk + 1) * K_bar[ii, kk] * Kinv[kk, jj]
                Tr_h2[ii, jj] += K_bar[ii, kk] * Kinv_check[kk, jj]
            if jj != 1 and jj != 2:
                # no reflection
                Mat[ii, 0] += T_h1[ii, jj] * y_tmp[jj] # h1
                Mat[ii, 1] += T_h2[ii, jj] * y_tmp[jj] # h2
                # with reflection
                Matr[ii, 0] += Tr_h1[ii, jj] * y_tmp[jj]  # h1
                Matr[ii, 1] += Tr_h2[ii, jj] * y_tmp[jj]  # h2
            if jj == 1:
                # no reflection
                Mat[ii, 2] += T_h1[ii, jj] # h1 * u
                Mat[ii, 4] += T_h2[ii, jj] # h2 * u
                # with reflection
                Matr[ii, 2] += Tr_h1[ii, jj]  # h1 * u
                Matr[ii, 4] += Tr_h2[ii, jj]  # h2 * u
            if jj == 2:
                # no reflection
                Mat[ii, 3] += T_h1[ii, jj] # h1 * v
                Mat[ii, 5] += T_h2[ii, jj] # h2 * v
                # with reflection
                Matr[ii, 3] += Tr_h1[ii, jj]  # h1 * v
                Matr[ii, 5] += Tr_h2[ii, jj]  # h2 * v

    # removing the rows corresponding to unknown y_bar
    Mat = np.delete(Mat, [1, 2], axis=0)
    Matr = np.delete(Matr, [1, 2], axis=0)

    # collecting known values of y_bar
    r = np.vstack((np.array([[y11_bar_est], [y22_bar_est]]), vectorize(Y2_bar_est)))

    if method == 'least_squares':
        x = cp.Variable((Mat.shape[1], 1))
        objective = cp.Minimize(cp.sum_squares(Mat @ x - r))
        constraints = [cp.norm(x) <= 10]
        prob = cp.Problem(objective, constraints)

        res_cp = prob.solve()
        sol = np.copy(x.value)
        h1_val = sol[0, 0]
        h2_val = sol[1, 0]
        u_1 = sol[2, 0] / h1_val
        v_1 = sol[3, 0] / h1_val
        u_2 = sol[4, 0] / h2_val
        v_2 = sol[5, 0] / h2_val

        print("\n Lyap-like least square errors closed form solution")
        print(np.array([[h1_val, h2_val], [u_1, u_2], [v_1, v_2]]))

    elif method == 'least_squares_scipy':
        ub = np.array([1, 1, np.infty, np.infty, np.infty, np.infty])
        lb = -1 * ub
        n_meas = (N - M) * M + M
        res = lsq_linear(Mat, r.reshape(n_meas), bounds=(lb, ub))
        h1_val = res.x[0]
        h2_val = res.x[1]
        u_1 = res.x[2] / h1_val
        v_1 = res.x[3] / h1_val
        u_2 = res.x[4] / h2_val
        v_2 = res.x[5] / h2_val

        print("\n Lyap-like least square errors from scipy lsq_linear")
        print(np.array([[h1_val, h2_val], [u_1, u_2], [v_1, v_2]]))

    elif method == 'scipy_minimize':
        '''
            scipy.optimize.minimize
        '''
        tol1 = 1e-3
        tol2 = 1e-3
        con1 = lambda x: x[1] * x[2] - x[0] * x[4]
        con2 = lambda x: x[1] * x[3] - x[0] * x[5]
        con3 = lambda x: x[0] ** 2 + x[1] ** 2 - 1
        nlc1 = NonlinearConstraint(con1, -tol1, tol1)
        nlc2 = NonlinearConstraint(con2, -tol1, tol1)
        # nlc3 = NonlinearConstraint(con3, -tol2, tol2)
        cons = (nlc1, nlc2)
        res = minimize(func, np.zeros(6), args=(Mat, r), constraints=cons)
        resr = minimize(func, np.zeros(6), args=(Matr, r), constraints=cons)
        if res.fun < resr.fun:
            sol = res
        else:
            sol = resr

        h1_val = sol.x[0]
        h2_val = sol.x[1]
        u_1 = sol.x[2] / h1_val
        v_1 = sol.x[3] / h1_val
        u_2 = sol.x[4] / h2_val
        v_2 = sol.x[5] / h2_val

        print("\n Lyap-like least square errors from scipy minimize")
        print(np.array([[h1_val, h2_val], [u_1, u_2], [v_1, v_2]]))
        print("\n Cost function")
        print(sol.fun)
    elif method == 'total_least_squares':
        r_tls, Mat_tls, x_tls, frob = tls(Mat, r)

        h1_val = x_tls[0, 0]
        h2_val = x_tls[1, 0]
        u_1 = x_tls[2, 0] / h1_val
        v_1 = x_tls[3, 0] / h1_val
        u_2 = x_tls[4, 0] / h2_val
        v_2 = x_tls[5, 0] / h2_val

        print("\n Lyap-like least square errors closed form solution")
        print(np.array([[h1_val, h2_val], [u_1, u_2], [v_1, v_2]]))
    else:
        print('Method not recognized!')

    Y_est = np.hstack((np.array([[y11_est, v_1], [u_1, y22_est]]), Y2_est))
    Y_est2 = np.hstack((np.array([[y11_est, v_2], [u_2, y22_est]]), Y2_est))
    X_est = np.linalg.pinv(UA).dot(Y_est).dot(np.linalg.pinv(VA))
    X_est2 = np.linalg.pinv(UA).dot(Y_est2).dot(np.linalg.pinv(VA))
    H_est = np.array([[h1_val, -h2_val], [h2_val, h1_val]])

    return X_est, H_est

def swap_rows(M, N):
    swap_idx = np.zeros(M * N, dtype=int)
    for ii in range(N):
        swap_idx[M * ii] = M * ii + 1
        swap_idx[M * ii + 1] = M * ii

    return swap_idx

def func(x, A, b):
    z = np.array([x[0], x[1], x[2], x[3], x[4], x[5]]).reshape((6, 1))

    tmp = (A.dot(z) - b).reshape((b.shape[0], 1))
    tmp2 = tmp.transpose().dot(tmp)
    cost = tmp2[0][0] #+ x[6] * (x[1] * x[2] - x[0] * x[4]) + x[7] * (x[1] * x[3] - x[0] * x[5])

    return cost

def elim_mat(m):
    T = np.tril(np.ones(m)) # Lower triangle of 1's
    f = np.nonzero(vectorize(T).squeeze()) # Get linear indexes of 1's
    k = int(m* (m + 1.) / 2.) # Row size of L
    m2 = m * m # Colunm size of L
    # L = np.zeros((m2, k)) # Start with L'
    L = np.zeros(m2 * k)  # Start with L'
    x = f[0] + m2 * np.arange(0, k) # Linear indexes of the 1's within L'
    L[x] = 1 # Put the 1's in place
    L = vectorize_inverse(L, rows=m2, cols=k)
    L = L.T # Now transpose to actual L

    return L

def polygon(X):
    D, N = X.shape
    c = np.mean(X, 1) # mean / central point
    d = X - c[:, None] # vectors connecting the central point and the given points
    th = np.zeros(N)
    th = np.arctan2(d[1, :], d[0, :])  # angle above x axis
    # for jj in range(X.shape[1]):
    #     th[jj] = np.arctan2(d[1, jj], d[0, jj]) # angle above x axis
    #     print(th)
    idx = np.argsort(th) # sorting the angles
    Y = X[:, idx] # sorting the given points
    Y = np.hstack((Y, Y[:, 0].reshape(D, 1))) # add the first at the end to close the polygon

    return Y

# source: https://github.com/RyotaBannai/total-least-squares/blob/master/tsl.py
def tls(X, y):
    if len(X.shape) is 1:
        n = 1
        X = X.reshape(len(X), 1)
    else:
        n = np.array(X).shape[1]  # the number of variable of X

    Z = np.vstack((X.T, y.T)).T
    U, s, Vt = svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = - Vxy / Vyy  # total least squares soln

    np.linalg.solve()

    Xtyt = - Z.dot(V[:, n:]).dot(V[:, n:].T)
    Xt = Xtyt[:, :n]  # X error
    y_tls = (X + Xt).dot(a_tls)

    fro_norm = norm(Xtyt, 'fro')  # Frobenius norm

    return y_tls, X + Xt, a_tls, fro_norm

def procrustes_error(Z, Z_bar):
    H, scale = orthogonal_procrustes(Z.T, Z_bar.T)
    Z_proc = Z.T @ H
    err_z = vectorize(Z_bar - Z_proc.T)

    return np.squeeze(err_z), H