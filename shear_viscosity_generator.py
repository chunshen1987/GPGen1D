from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PCATransformation import PCATransformation

def transformation(eta_s):
    eta_s_max = 0.3
    scale = 1.0
    return eta_s_max/2.*(1. + np.tanh(scale*eta_s))

def eta_s_file_writer(T, eta_s, filename):
    """
        This function writes the eta_s to a pickle file with a dictionary 
        for each eta_s. The different columns are: T, eta_s
    """
    eta_s_dict = {}
    for es in range(len(eta_s)):
        data = np.column_stack((T, eta_s[es]))
        eta_s_dict[f'{es:04}'] = data
    with open(filename, 'wb') as f:
        pickle.dump(eta_s_dict, f)

def main(ranSeed: int, number_of_eta_s: int) -> None:
    # print out the minimum and maximum values of the training data x_values
    T_min = 0.0
    T_max = 0.5
    print(f"Minimum of the datapoints is: {T_min}")
    print(f"Maximum of the datapoints is: {T_max}")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()


    T_GP = np.linspace(T_min, T_max, 100).reshape(-1, 1)
    T_plot = T_GP.flatten()

    correlation_length_min = 0.05
    correlation_length_max = 0.20

    eta_s_set = []
    nsamples_per_batch = max(1, int(number_of_eta_s/100))
    progress = 0
    while len(eta_s_set) < number_of_eta_s:
        correlation_length = np.random.uniform(correlation_length_min,
                                               correlation_length_max)
        print(f"Progress {progress}%, corr len = {correlation_length:.2f} ...")
        kernel = RBF(length_scale=correlation_length, length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        eta_s_vs_T_GP = gpr.sample_y(T_plot.reshape(-1, 1), 
                                     n_samples=nsamples_per_batch, 
                                     random_state=randomness).transpose()
        eta_s_vs_T_GP = transformation(eta_s_vs_T_GP)
        for sample_i in eta_s_vs_T_GP:
            eta_s_set.append(sample_i)
        progress += 1

    # make verification plots
    plt.figure()
    for i in range(number_of_eta_s):
        if i%(nsamples_per_batch*5) == 0:
            plt.plot(T_plot, eta_s_set[i], '-')

    plt.xlim([T_min, T_max])
    plt.ylim([0, 0.45])
    plt.xlabel(r"T [GeV]")
    plt.ylabel(r"$\eta/s$")
    plt.tight_layout()
    plt.savefig(f"eta_s_samples.png", dpi=600)
    plt.clf()

    # Compute the 90% prior for the shear viscosity from the eta_s_set for each T
    eta_s_set = np.array(eta_s_set)
    low, high = np.percentile(eta_s_set, [0, 100.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='b', label='100% prior')
    plt.plot(T_plot, high, 'b-', lw=1)
    plt.plot(T_plot, low, 'b-', lw=1)
    low, high = np.percentile(eta_s_set, [50-90/2., 50+90/2.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='g', label='90% prior')
    plt.plot(T_plot, high, 'g-', lw=1)
    plt.plot(T_plot, low, 'g-', lw=1)
    low, high = np.percentile(eta_s_set, [50-50/2., 50+50/2.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='r', label='50% prior')
    plt.plot(T_plot, high, 'r-', lw=1)
    plt.plot(T_plot, low, 'r-', lw=1)
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\eta/s$")
    plt.xlim([T_min, T_max])
    plt.ylim([0, 0.45])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"eta_s_prior.png")
    plt.clf()

    # write the EoS to a file
    eta_s_file_writer(T_GP, eta_s_set, f"eta_s.pkl")

    # check PCA
    plt.figure()
    varianceList = [0.9, 0.95, 0.99]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)
        eta_s_set = np.array(eta_s_set)
        scaled = scaler.fit_transform(eta_s_set)
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        zeta_s_reconstructed = PCA_fitted.inverse_transform(PCs)
        zeta_s_reconstructed = scaler.inverse_transform(zeta_s_reconstructed)

        RMS_error = np.sqrt(
            np.mean((eta_s_set - zeta_s_reconstructed)**2, axis=0))
        plt.plot(T_plot, RMS_error,
                 label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
    plt.legend()
    plt.xlim([T_min, T_max])
    plt.ylim([0, 0.1])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\eta/s$ RMS error")
    plt.savefig("RMS_errors_shear_viscosity.png")
    plt.clf()

    # check the distribution for PCs
    pca = PCATransformation(0.97)
    PCs = pca.fit_transform(eta_s_set)
    print(f"Number of components = {PCs.shape[1]}")
    print(PCs.min(axis=0), PCs.max(axis=0))
    for i in range(PCs.shape[1]):
        plt.figure()
        plt.hist(PCs[:, i], bins=17, density=True)
        plt.savefig(f"PC{i}.png")
        plt.clf()

    with open("shearPCA.pickle", "wb") as f:
        pickle.dump(pca, f)

    with open("shearPCAChain.pickle", "wb") as f:
        pickle.dump(PCs, f)

if __name__ == "__main__":
    ranSeed = 23
    number_of_eta_s = 10000
    main(ranSeed, number_of_eta_s)
