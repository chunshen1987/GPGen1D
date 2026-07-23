from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
from PCATransformation import PCATransformation

ACCURACY = 1e-6
MAXITER  = 100


def transform_e(ee):
    expon = 1./1.5
    a = 0.25
    e0 = 1.0  # GeV/fm^3
    loge = np.log(ee/e0)
    return np.sign(loge)*((a*np.abs(loge))**expon)


def inverse_transform_e(ee):
    expon = 1./1.5
    a = 0.25
    e0 = 1.0  # GeV/fm^3
    sign = np.sign(ee)
    ee = np.abs(ee)
    return e0 * np.exp(sign*ee**(1./expon)/a)


def transfrom_cs2_to_GP(cs2):
    return np.arctanh(6 * cs2 - 1)


def transfrom_GP_to_cs2(GP):
    return (1/6) * (1 + np.tanh(GP))


def compute_EOS(ed_GP, cs2_GP, hrgEOS_tb, e0):
    e_tb = (np.linspace(ed_GP[0]**0.25, ed_GP[-1]**0.25, 1000))**4
    cs2_tb = np.interp(e_tb**0.25, ed_GP**0.25, cs2_GP)
    p_tb = np.zeros(len(e_tb))
    s_tb = np.zeros(len(e_tb))
    idx = (e_tb <= e0)
    p_tb[idx] = np.interp(e_tb[idx], hrgEOS_tb[:, 0], hrgEOS_tb[:, 1])
    s_tb[idx] = np.interp(e_tb[idx], hrgEOS_tb[:, 0], hrgEOS_tb[:, 2])
    idx0 = np.where(e_tb > e0)[0][0]
    idx = e_tb > e0
    s0 = np.interp(e_tb[idx0], hrgEOS_tb[:, 0], hrgEOS_tb[:, 2])
    for i in range(idx0, len(e_tb)):
        de = e_tb[i] - e_tb[i-1]
        p_tb[i] = p_tb[i-1] + (cs2_tb[i] + cs2_tb[i-1]) / 2. * de
        if i > idx0:
            s_tb[i] = s_tb[i-1] + (1. / (e_tb[i] + p_tb[i]) + 1. / (e_tb[i-1] + p_tb[i-1])) / 2 * de
    s_tb[idx] = s0 * np.exp(s_tb[idx])
    T_tb = (e_tb + p_tb) / s_tb
    return np.array([e_tb, p_tb, s_tb, T_tb, cs2_tb]).T


def EoS_file_writer(eos_set, filename):
    """
        This function writes the EoS to a pickle file with a dictionary 
        for each EoS. The different columns are: e, P, T
    """
    neos, ne, nvar = eos_set.shape
    EoS_dict = {}
    for ieos in range(min(1000, neos)):
        EoS_dict[f'{ieos:04}'] = eos_set[ieos][:, [0, 1, 3]].astype(np.float32)
    with open(filename, 'wb') as f:
        pickle.dump(EoS_dict, f)


def main(ranSeed: int, number_of_EoS: int, min_e_mask_region: float, 
         use_anchor_point: bool, anchor_point) -> None:
    # load the full EOS table for verification
    hotQCDEoS = np.fromfile("hrg_hotqcd_eos_binary.dat").reshape(-1, 4)
    hotQCD_e = (hotQCDEoS[1:, 0] + hotQCDEoS[:-1, 0]) / 2
    hotQCD_cs2 = ((hotQCDEoS[1:, 1] - hotQCDEoS[:-1, 1])
                  / (hotQCDEoS[1:, 0] - hotQCDEoS[:-1, 0]))
    hrgEOS = np.loadtxt("HRGEOS_PST-urqmd_v3.3+.dat")

    e0 = min_e_mask_region
    P0 = np.interp(e0, hotQCDEoS[:, 0], hotQCDEoS[:, 1])
    s0 = np.interp(e0, hotQCDEoS[:, 0], hotQCDEoS[:, 2])

    # mask for the training data and exclude the region where the T 
    # (first column) is between min_T_mask_region and max_T_mask_region
    mask = (hrgEOS[:, 0] < min_e_mask_region)
    training_data = np.array([transform_e(hrgEOS[mask, 0]),
                              transfrom_cs2_to_GP(hrgEOS[mask, 4])]).T

    # add an anchor point to the training data
    if use_anchor_point:
        anchor_point = np.array([transform_e(anchor_point[:, 0]),
                                 transfrom_cs2_to_GP(anchor_point[:, 1])]).T
        print(training_data.shape, anchor_point.shape)
        training_data = np.vstack((training_data, anchor_point))
        print(f"Anchor point added: {anchor_point}")
        print(training_data.shape)

    # print out the minimum and maximum values of the training data x_values
    e_min = np.min(training_data[:, 0])  # the min of actual data points
    e_max = transform_e(3e3)
    print(f"Minimum of the datapoints is: {e_min}")
    print(f"Maximum of the datapoints is: {e_max}")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()

    # define GP kernel
    #kernel = 1 * RBF(length_scale=0.05, length_scale_bounds=(1e-2, 10))
    kernel = 1 * Matern(length_scale=0.05, length_scale_bounds=(1e-2, 10),
                        nu=3/2)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)

    x_train = training_data[:, 0].reshape(-1, 1)
    gpr.fit(x_train, training_data[:, 1])
    print(f"GP score: {gpr.score(x_train, training_data[:, 1])}")
    print(gpr.kernel_)
    etilde_GP = np.linspace(e_min, e_max, 1000).reshape(-1, 1)

    EOS_set = []

    cs2_GP = transfrom_GP_to_cs2(
            gpr.sample_y(etilde_GP, number_of_EoS,
                         random_state=randomness).transpose())

    #eos_test = compute_EOS(hotQCD_e, hotQCD_cs2, hotQCDEoS, min_e_mask_region)
    #for j in range(1000):
    #    if eos_test[j, 0] > 0.1 and eos_test[j, 0] < 1.0:
    #        print(eos_test[j, 0], eos_test[j, 1],
    #              np.interp(eos_test[j, 0], hotQCDEoS[:, 0], hotQCDEoS[:, 1]))
    #        print(eos_test[j, 0], eos_test[j, 2],
    #              np.interp(eos_test[j, 0], hotQCDEoS[:, 0], hotQCDEoS[:, 2]))
    #        print(eos_test[j, 0], eos_test[j, 3],
    #              np.interp(eos_test[j, 0], hotQCDEoS[:, 0], hotQCDEoS[:, 3]))
    #exit(0)

    ee_GP = inverse_transform_e(etilde_GP.flatten())
    for i in range(number_of_EoS):
        EOS_set.append(compute_EOS(ee_GP, cs2_GP[i, :],
                                   hrgEOS, min_e_mask_region))
        #if i == 23:
        #    for j in range(1000):
        #        if EOS_set[i][j, 0] > 0.1 and EOS_set[i][j, 0] < 1.0:
        #            print(EOS_set[i][j, :])
        #    exit(0)
    EOS_set = np.array(EOS_set)

    # make verification plots
    plt.figure()
    cs2_low, cs2_high = np.percentile(cs2_GP, [50-95/2., 50+95/2.], axis=0)
    plt.fill_between(np.log(ee_GP), cs2_low, cs2_high,
                     color='g', alpha=0.2, label="95% CI")
    cs2_low, cs2_high = np.percentile(cs2_GP, [50-90/2., 50+90/2.], axis=0)
    plt.fill_between(np.log(ee_GP), cs2_low, cs2_high,
                     color='b', alpha=0.2, label="90% CI")
    cs2_low, cs2_high = np.percentile(cs2_GP, [50-70/2., 50+70/2.], axis=0)
    plt.fill_between(np.log(ee_GP), cs2_low, cs2_high,
                     color='r', alpha=0.2, label="70% CI")
    for i in range(20):
        plt.plot(np.log(ee_GP), cs2_GP[i, :], 'k-')
    plt.scatter(np.log(hotQCD_e), hotQCD_cs2,
                marker='+', color='b', s=20, label="lattice EOS")
    plt.scatter(np.log(inverse_transform_e(training_data[:, 0])),
                transfrom_GP_to_cs2(training_data[:, 1]),
                marker='x', color='r', s=20, label="training data")

    plt.legend()
    plt.xlim([-5.0, 8.0])
    plt.ylim([0, 1./3.])
    plt.xlabel(r"$log(e/e_0)$")
    plt.ylabel(r"$c_s^2$")
    plt.show()

    # plot cs2 vs e
    plt.figure()
    cs2_low, cs2_high = np.percentile(
        EOS_set[:, :, 4], [50-95/2., 50+95/2.], axis=0)
    plt.fill_between(EOS_set[0, :, 0], cs2_low, cs2_high,
                     color='g', alpha=0.2, label="95% CI")
    cs2_low, cs2_high = np.percentile(
        EOS_set[:, :, 4], [50-90/2., 50+90/2.], axis=0)
    plt.fill_between(EOS_set[0, :, 0], cs2_low, cs2_high,
                     color='b', alpha=0.2, label="90% CI")
    cs2_low, cs2_high = np.percentile(
        EOS_set[:, :, 4], [50-70/2., 50+70/2.], axis=0)
    plt.fill_between(EOS_set[0, :, 0], cs2_low, cs2_high,
                     color='r', alpha=0.2, label="70% CI")
    for i in range(20):
        plt.plot(EOS_set[i, :, 0], EOS_set[i, :, 4], '-k', zorder=1)
    plt.scatter(hotQCD_e, hotQCD_cs2,
                marker='+', color='b', s=20, label="lattice EOS")

    plt.legend()
    plt.xscale('log')
    plt.xlim([0.05, 1000.])
    plt.ylim([0., 1./3.])
    plt.xlabel(r"$e$ (GeV/fm$^3$)")
    plt.ylabel(r"$c_s^2$")
    plt.savefig("cs2_prior.png")
    plt.show()

    # plot P/e vs e
    plt.figure()
    Povere_low, Povere_high = np.percentile(
        EOS_set[:, :, 1]/EOS_set[0, :, 0], [50-95/2., 50+95/2.], axis=0)
    plt.fill_between(EOS_set[0, :, 0], Povere_low, Povere_high,
                     color='g', alpha=0.2, label="95% CI")
    for i in range(20):
        plt.plot(EOS_set[i, :, 0], EOS_set[i, :, 1]/EOS_set[i, :, 0],
                 '-k', zorder=1)
    plt.scatter(hotQCDEoS[:, 0], hotQCDEoS[:, 1]/hotQCDEoS[:, 0],
                marker='+', color='b', s=20, label="lattice EOS")

    plt.xscale('log')
    plt.xlim([0.05, 1000.])
    plt.xlabel(r"$e$ (GeV/fm$^3$)")
    plt.ylabel(r"$P/e$")
    plt.show()

    # plot s/e^{3/4} vs e
    hbarc = 0.197327    # GeV/fm
    plt.figure()
    sovere_low, sovere_high = np.percentile(
        EOS_set[:, :, 2]/(EOS_set[0, :, 0]/hbarc)**0.75,
        [50-95/2., 50+95/2.], axis=0)
    plt.fill_between(EOS_set[0, :, 0], sovere_low, sovere_high,
                     color='g', alpha=0.2, label="95% CI")
    for i in range(20):
        plt.plot(EOS_set[i, :, 0],
                 EOS_set[i, :, 2]/(EOS_set[i, :, 0]/hbarc)**0.75,
                 '-k', zorder=1)
    plt.scatter(hotQCDEoS[:, 0], hotQCDEoS[:, 2]/(hotQCDEoS[:, 0]/hbarc)**0.75,
                marker='+', color='b', s=20, label="lattice EOS")

    plt.xscale('log')
    plt.xlim([0.05, 1000.])
    plt.xlabel(r"$e$ (GeV/fm$^3$)")
    plt.ylabel(r"$s/e^{3/4}$")
    plt.show()

    # plot T vs e
    plt.figure()
    T_low, T_high = np.percentile(
        EOS_set[:, :, 3], [50-95/2., 50+95/2.], axis=0)
    plt.fill_between(EOS_set[0, :, 0], T_low, T_high,
                     color='g', alpha=0.2, label="95% CI")
    for i in range(20):
        plt.plot(EOS_set[i, :, 0], EOS_set[i, :, 3], '-k', zorder=1)
    plt.scatter(hotQCDEoS[:, 0], hotQCDEoS[:, 3],
                marker='+', color='b', s=20, label="lattice EOS")

    plt.xscale('log')
    plt.xlim([0.05, 1000.])
    plt.xlabel(r"$e$ (GeV/fm$^3$)")
    plt.ylabel(r"$T$ (GeV)")
    plt.show()

    # plot cs2 vs T
    plt.figure()
    for i in range(20):
        plt.plot(EOS_set[i, :, 3], EOS_set[i, :, 4], '-k', zorder=1)
    plt.scatter(hotQCDEoS[:-1, 3], hotQCD_cs2,
                marker='+', color='b', s=20, label="lattice EOS")

    plt.xlim([0.05, 1.])
    plt.xlabel(r"$T$ (GeV)")
    plt.ylabel(r"$c_s^2$")
    plt.show()

    # write the EoS to a file
    EoS_file_writer(EOS_set, f"EoS.pkl")

    # check PCA
    plt.figure()
    varianceList = [0.7, 0.9, 0.95, 0.99]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)
        scaled = scaler.fit_transform(EOS_set[:, :, 4])  # cs2
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        cs2_reconstructed = PCA_fitted.inverse_transform(PCs)
        cs2_reconstructed = scaler.inverse_transform(cs2_reconstructed)

        RMS_error = np.sqrt(np.mean(
            ((EOS_set[:, :, 4] - cs2_reconstructed)/EOS_set[:, :, 4])**2,
            axis=0))
        plt.plot(EOS_set[0, :, 0], RMS_error,
                 label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
    plt.legend()
    plt.xscale('log')
    plt.xlim([0.05, 1000.])
    plt.ylim([0, None])
    plt.xlabel(r"$e$ [GeV/fm$^3$]")
    plt.ylabel(r"$c_s^2$ RMS relative error")
    plt.savefig("RMS_errors_cs2.png")
    plt.show()
    plt.clf()

    # print out the PCs ranges
    pca = PCATransformation(0.9)
    PCs = pca.fit_transform(EOS_set[:, :, 4])
    print(f"Number of components = {PCs.shape[1]}")
    print(PCs.min(axis=0), PCs.max(axis=0))
    for i in range(PCs.shape[1]):
        plt.figure()
        plt.hist(PCs[:, i], bins=17, density=True)
        plt.savefig(f"PC{i}.png")
        plt.clf()

    with open("EoSPCA.pickle", "wb") as f:
        pickle.dump(pca, f)

    with open("EoSPCAChain_training.pickle", "wb") as f:
        pickle.dump(PCs[:int(0.8*number_of_EoS), :], f)
    with open("EoSPCAChain_validation.pickle", "wb") as f:
        pickle.dump(PCs[int(0.8*number_of_EoS):, :], f)

if __name__ == "__main__":
    ranSeed = 23
    number_of_EoS = 100000
    min_e_mask_region = 0.20
    use_anchor_point = True
    anchor_point = np.array([1e5, 0.33]).reshape(-1, 2)   # anchor point for the GP (e, cs^2)
    main(ranSeed, number_of_EoS, min_e_mask_region,
         use_anchor_point, anchor_point)
