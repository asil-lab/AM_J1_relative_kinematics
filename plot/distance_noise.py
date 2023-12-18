import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
font = {'size': 12}
mpl.rc('font', **font)

SEED = 2110
SNR = np.array([-30, -25, -20, -15, -10, -5, 0])
d_array = np.array([1.0, 10., 100.])
K = 1000
diff_d = np.zeros((len(d_array), len(SNR)))
# percent_err = np.zeros(len(SNR))
for kk in range(len(SNR)):
    for ll in range(len(d_array)):
        d = d_array[ll]
        eta = d * 10. ** (SNR[kk] / 10.)
        noise = np.random.normal(0.0, eta, K)
        d_full = (d + noise) ** 2
        d_approx = d ** 2 + 2 * d * noise
        diff_d[ll, kk] = np.sqrt(np.linalg.norm(d_full - d_approx) ** 2 / K)
        # percent_err[kk] = np.mean(np.abs(d_full - d_approx) / d_full) * 100

fig, axs = plt.subplots(1, 1)
axs.set_yscale('log')
# axs2 = axs.twinx()
axs.plot(-SNR, diff_d[0, :], 'b.--', lw=2., ms=10., label='d = ' + str(int(d_array[0])) + ' m')
axs.plot(-SNR, diff_d[1, :], 'rd--', lw=2., ms=10., label='d = ' + str(int(d_array[1])) + ' m')
axs.plot(-SNR, diff_d[2, :], 'kx--', lw=2., ms=10., label='d = ' + str(int(d_array[2])) + ' m')
axs.set_xlabel(r'SNR [dB]')
axs.set_ylabel('RMSE [$m^2$]')
axs.legend()
# axs2.plot(-SNR, percent_err, 'rx-')
axs.grid()
plt.show()

print('finished!')