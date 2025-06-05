from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PCATransformation import PCATransformation

def transformation(zeta_s):
    zeta_s_max = 0.3
    scale = 1.0
    return zeta_s_max/2.*(1. + np.tanh(scale*zeta_s))


def zeta_s_file_writer(T, zeta_s, filename) -> None:
    """ This function writes the zeta_s to a pickle file with a dictionary
        for each zeta_s. The different columns are: T, zeta_s
    """
    zeta_s_dict = {}
    for zs in range(len(zeta_s)):
        data = np.column_stack((T, zeta_s[zs]))
        zeta_s_dict[f'{zs:04}'] = data
    with open(filename, 'wb') as f:
        pickle.dump(zeta_s_dict, f)


def main(ranSeed: int, number_of_zeta_s: int) -> None:
    T_min = 0.00
    T_max = 0.50
    print(f"Minimum of the datapoints is: {T_min}")
    print(f"Maximum of the datapoints is: {T_max}")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()

    Tlow = np.array([T_min])
    Thigh = np.array([T_max])
    expon_low = np.array([-3])
    expon_high = np.array([-3])

    T_GP = np.concatenate((Tlow, Thigh))
    training_data = np.concatenate((expon_low, expon_high))

    T_plot = np.linspace(T_min, T_max, 100)

    correlation_length_min = 0.05
    correlation_length_max = 0.15

    zeta_s_set = []

    nsamples_per_batch = max(1, int(number_of_zeta_s/100))
    progress = 0
    while len(zeta_s_set) < number_of_zeta_s:
        correlation_length = np.random.uniform(correlation_length_min,
                                               correlation_length_max)
        print(f"Progress {progress}%, corr len = {correlation_length:.2f} ...")
        kernel = RBF(length_scale=correlation_length,
                     length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        gpr.fit(T_GP.reshape(-1, 1), training_data)
        zeta_s_vs_T_GP = gpr.sample_y(T_plot.reshape(-1, 1),
                                      nsamples_per_batch,
                                      random_state=randomness).transpose()
        zeta_s_vs_T_GP = transformation(zeta_s_vs_T_GP)
        for sample_i in zeta_s_vs_T_GP:
            zeta_s_set.append(sample_i)
        progress += 1

    # make verification plots
    plt.figure()
    plt.scatter(T_GP, transformation(training_data))
    for i in range(number_of_zeta_s):
        if i%(nsamples_per_batch*5) == 0:
            plt.plot(T_plot, zeta_s_set[i], '-')

    plt.xlim([T_min, T_max])
    plt.ylim([0, 0.42])
    plt.xlabel(r"T [GeV]")
    plt.ylabel(r"$\zeta/s$")
    #plt.yscale("log")
    #plt.show()
    plt.tight_layout()
    plt.savefig(f"zeta_s_samples.png")
    plt.clf()

    # Compute the 90% prior for the bulk viscosity from the zeta_s_set for each T
    zeta_s_set = np.array(zeta_s_set)
    low, high = np.percentile(zeta_s_set, [0, 100.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='b', label='100% prior')
    plt.plot(T_plot, high, 'b-', lw=1)
    plt.plot(T_plot, low, 'b-', lw=1)
    low, high = np.percentile(zeta_s_set, [50-90/2., 50+90/2.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='g', label='90% prior')
    plt.plot(T_plot, high, 'g-', lw=1)
    plt.plot(T_plot, low, 'g-', lw=1)
    low, high = np.percentile(zeta_s_set, [50-50/2., 50+50/2.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='r', label='50% prior')
    plt.plot(T_plot, high, 'r-', lw=1)
    plt.plot(T_plot, low, 'r-', lw=1)
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\zeta/s$")
    plt.xlim([T_min, T_max])
    plt.ylim([0, 0.42])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"zeta_s_prior.png")
    plt.clf()

    # write the EoS to a file
    zeta_s_file_writer(T_plot, zeta_s_set, f"zeta_s.pkl")

    # check PCA
    plt.figure()
    varianceList = [0.9, 0.95, 0.99]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)
        zeta_s_set = np.array(zeta_s_set)
        scaled = scaler.fit_transform(zeta_s_set)
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        zeta_s_reconstructed = PCA_fitted.inverse_transform(PCs)
        zeta_s_reconstructed = scaler.inverse_transform(zeta_s_reconstructed)

        RMS_error = np.sqrt(
            np.mean((zeta_s_set - zeta_s_reconstructed)**2, axis=0))
        plt.plot(T_plot, RMS_error,
                 label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
    plt.legend()
    plt.xlim([T_min, T_max])
    plt.ylim([0, 0.05])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\zeta/s$ RMS error")
    plt.savefig("RMS_errors_bulk_viscosity.png")
    plt.clf()

    #plt.figure()
    #idxArr = np.random.choice(len(zeta_s_set), 3)
    #for idx in idxArr:
    #    plt.plot(T_plot, zeta_s_set[idx], '-')
    #    plt.plot(T_plot, zeta_s_reconstructed[idx], '--')
    #plt.show()

    # check the distribution for PCs
    pca = PCATransformation(0.95)
    PCs = pca.fit_transform(zeta_s_set)
    print(f"Number of components = {PCs.shape[1]}")
    print(PCs.min(axis=0), PCs.max(axis=0))
    for i in range(PCs.shape[1]):
        plt.figure()
        plt.hist(PCs[:, i], bins=17, density=True)
        plt.savefig(f"PC{i}.png")
        plt.clf()

    with open("bulkPCA.pickle", "wb") as f:
        pickle.dump(pca, f)



if __name__ == "__main__":
    ranSeed = 23
    number_of_zeta_s = 2000
    main(ranSeed, number_of_zeta_s)
