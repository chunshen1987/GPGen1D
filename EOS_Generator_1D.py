from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
from PCATransformation import PCATransformation

ACCURACY = 1e-6
MAXITER  = 100

def compute_derivative(x, y):
    """This function compute dy/dx using finite difference"""
    return np.gradient(y, x, edge_order=2)

def compute_energy_density(T, P):
    """This function computes energy density"""
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P    # energy density
    return e

def compute_entropy_density(T, P):
    """This function computes entropy density"""
    dPdT = compute_derivative(T, P)
    s = dPdT    # entropy density
    return s

def compute_speed_of_sound_square(T, P):
    """This function computes the speed of sound square"""
    e = compute_energy_density(T, P)
    dPde = compute_derivative(e, P)
    return dPde

def derivative_filter(x, y) -> bool:
    """
        This filter check whether the derivative is larger than 0
        for all array elements
    """
    dydx = compute_derivative(x, y)
    indices = dydx < 0.
    negative_derivatives = x[indices]

    if len(negative_derivatives) == 0:
        return True
    else:
        return False

def speed_sound_squared_filter(T, P) -> bool:
    """
    	This filter ensures the speed of sound square is between 0 and 0.5.
    	The upper bound is chosen such that the causality constraints in
    	the hydrodynamical simulations are satisfied.
    """
    cs2Max = 1./3.
    cs2 = compute_speed_of_sound_square(T, P)
    index = (cs2 > 0.) & (cs2 < cs2Max)
    physical_points = cs2[index]

    if len(cs2) == len(physical_points):
        return True
    else:
        return False

def is_a_physical_eos(T, P) -> bool:
    """
        This calls the different physics filters to check if the 
        EoS is a physical one
    """
    if not derivative_filter(T, P):
        return False

    dPdT = compute_derivative(T, P)
    if not derivative_filter(T, dPdT):
        return False

    if not speed_sound_squared_filter(T, P):
        return False

    return True

def binary_search_1d(y_local, f_y, x_min, x_max):
    """
        This function performs a binary search to find the x value
        that corresponds to the y value y_local
    """
    iteration = 0
    y_low = f_y(x_min)
    y_up = f_y(x_max)
    if y_local < y_low:
        return x_min
    elif y_local > y_up:
        return x_max
    else:
        x_mid = (x_max + x_min) / 2.
        y_mid = f_y(x_mid)
        abs_err = abs(y_mid - y_local)
        rel_err = abs_err / abs(y_mid + y_local + 1e-15)
        while (rel_err > ACCURACY and abs_err > ACCURACY*1e-2 
               and iteration < MAXITER):
            if y_local < y_mid:
                x_max = x_mid
            else:
                x_min = x_mid
            x_mid = (x_max + x_min) / 2.
            y_mid = f_y(x_mid)
            abs_err = abs(y_mid - y_local)
            rel_err = abs_err / abs(y_mid + y_local + 1e-15)
            iteration += 1
        return x_mid

def invert_EoS_tables(T, P):
    """
        This function inverts the EoS table to get e(T), it also computes the
        pressure P(T)
    """
    e = compute_energy_density(T, P)
    f_e = interpolate.interp1d(T, e, kind='cubic')
    f_p = interpolate.interp1d(T, P, kind='cubic')

    e_bounds = [np.max((1e-10, np.min(e))), np.max(e)]
    e_list = np.linspace(e_bounds[0]**0.25, e_bounds[1]**0.25, 200)**4

    T_from_e = []
    for e_local in e_list:
        T_local = binary_search_1d(e_local, f_e, T[0], T[-1])
        T_from_e.append(T_local)
    T_from_e = np.array(T_from_e)
    return (e_list**0.25, f_p(T_from_e), T_from_e)

def EoS_file_writer(e, P, T, filename):
    """
        This function writes the EoS to a pickle file with a dictionary 
        for each EoS. The different columns are: e, P, T
    """
    EoS_dict = {}
    for EoS in range(len(e)):
        data = np.column_stack((e[EoS], P[EoS], T[EoS]))
        EoS_dict[f'{EoS:04}'] = data
    with open(filename, 'wb') as f:
        pickle.dump(EoS_dict, f)

def main(ranSeed: int, number_of_EoS: int, min_T_mask_region: float, 
         max_T_mask_region: float, bLogFlag: bool, use_anchor_point: bool,
         anchor_point: tuple) -> None:
    # load the full EOS table for verification
    validation_data = np.loadtxt("EoS_hotQCD.dat")

    Nskip = 2
    validation_data = validation_data[::Nskip, :]
    # mask for the training data and exclude the region where the T 
    # (first column) is between min_T_mask_region and max_T_mask_region
    mask = ((validation_data[:, 0] < min_T_mask_region) | 
                (validation_data[:, 0] > max_T_mask_region))
    training_data = validation_data[mask]

    # add an anchor point to the training data
    if use_anchor_point:
        training_data = np.vstack((training_data, anchor_point))
        print(f"Anchor point added: {anchor_point}")
        print(training_data.shape)

    # print out the minimum and maximum values of the training data x_values
    T_min = np.min(training_data[:, 0])  # the min of actual data points
    T_max = 0.8
    print(f"Minimum of the datapoints is: {T_min}")
    print(f"Maximum of the datapoints is: {T_max}")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()

    # define GP kernel
    kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=5e-5)

    if bLogFlag:
        # train GP with log(T) vs. log(P/T^4) because all the quantities
        # are positive
        x_train = np.log(training_data[:, 0]).reshape(-1, 1)
        gpr.fit(x_train, np.log(training_data[:, 1]))
        print(f"GP score: {gpr.score(x_train, np.log(training_data[:, 1]))}")
        T_GP = np.linspace(np.log(T_min), np.log(T_max), 1000).reshape(-1, 1)
        T_plot = np.exp(T_GP.flatten())
    else:
        x_train = training_data[:, 0].reshape(-1, 1)
        gpr.fit(x_train, training_data[:, 1])
        print(f"GP score: {gpr.score(x_train, training_data[:, 1])}")
        T_GP = np.linspace(T_min, T_max, 1000).reshape(-1, 1)
        T_plot = T_GP.flatten()

    print(gpr.kernel_)

    EOS_set = []

    iSuccess = 0
    iter = 0
    nsamples_per_batch = 1000
    while iSuccess < number_of_EoS:
        PoverT4_GP = gpr.sample_y(T_GP, nsamples_per_batch,
                                  random_state=randomness).transpose()
        for sample_i in PoverT4_GP:
            if bLogFlag:
                P_GP = np.exp(sample_i)*(T_plot**4)       # convert to P
            else:
                P_GP = sample_i*(T_plot**4)       # convert to P
            if is_a_physical_eos(T_plot, P_GP):
                EOS_set.append(P_GP)
                iSuccess += 1
                if iSuccess == number_of_EoS:
                    break
        iter += nsamples_per_batch
        print(f"Sample success rate: {float(iSuccess)/iter:.3f}")

    # make verification plots
    '''
    plt.figure()
    plt.scatter(training_data[:, 0], training_data[:, 1],
                marker='x', color='r', s=20, label="training data")
    plt.scatter(validation_data[:, 0], validation_data[:, 1],
                marker='+', color='b', s=20, label="validation data")
    for i in range(number_of_EoS):
        plt.plot(T_plot, EOS_set[i]/T_plot**4, '-')

    plt.legend()
    plt.xlim([0, 1.])
    #plt.ylim([0, 4.5])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$P/T^{4}$")
    plt.show()

    # plot e vs T
    plt.figure()
    for i in range(number_of_EoS):
        e = compute_energy_density(T_plot, EOS_set[i])
        plt.plot(T_plot, e/T_plot**4, '-')

    plt.xlim([0, 1.])
    #plt.ylim([0, 15])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$e/T^4$")
    plt.show()

    # plot cs^2 vs T
    plt.figure()
    for i in range(number_of_EoS):
        cs2 = compute_speed_of_sound_square(T_plot, EOS_set[i])
        plt.plot(T_plot, cs2, '-')

    plt.xlim([0, 1.])
    plt.ylim([0, 0.6])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$c_s^2$")
    plt.show()
    '''

    # create file with EoS table for one EoS with evenly spaced T values
    write_EOS_table_for_plot = False
    if write_EOS_table_for_plot:
        EoS_chosen = 0
        # write T, P/T^4, e/T^4, s/T^3, cs^2 to a file
        data = np.column_stack((T_plot, EOS_set[EoS_chosen]/T_plot**4,
                compute_energy_density(T_plot, EOS_set[EoS_chosen])/T_plot**4,
                compute_entropy_density(T_plot, EOS_set[EoS_chosen])/T_plot**3,
                compute_speed_of_sound_square(T_plot, EOS_set[EoS_chosen])))
        np.savetxt(f"EoS{EoS_chosen}.dat", data)

    # invert the EoS tables
    e_list_EoS = []
    P_list_EoS = []
    T_list_EoS = []
    for i in range(number_of_EoS):
        if (i+1) % 100 == 0:
            print(f"Inverting EoS table {i+1}/{number_of_EoS}")
        e_list, P_list, T_list = invert_EoS_tables(T_plot, EOS_set[i])
        e_list_EoS.extend([e_list])
        P_list_EoS.extend([P_list])
        T_list_EoS.extend([T_list])

    # write the EoS to a file
    EoS_file_writer(e_list_EoS, P_list_EoS, T_list_EoS, f"EoS.pkl")

    # check PCA
    plt.figure()
    varianceList = [0.3, 0.5, 0.7, 0.9]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)
        EOS_set = np.array(EOS_set)
        scaled = scaler.fit_transform(EOS_set)
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        EOS_reconstructed = PCA_fitted.inverse_transform(PCs)
        EOS_reconstructed = scaler.inverse_transform(EOS_reconstructed)

        RMS_error = np.sqrt(
            np.mean(((EOS_set/T_plot**4) - (EOS_reconstructed/T_plot**4))**2, axis=0))
        plt.plot(T_plot, RMS_error,
                 label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
    plt.legend()
    plt.xlim([T_plot[0], T_plot[-1]])
    plt.ylim([0, None])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$P/T^4$ RMS error")
    plt.savefig("RMS_errors_EOS.png")
    plt.clf()

    # check PCA with speed of sound square
    plt.figure()
    varianceList = [0.3, 0.5, 0.7, 0.9]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)
        EOS_set = np.array(EOS_set)
        scaled = scaler.fit_transform(EOS_set)
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        EOS_reconstructed = PCA_fitted.inverse_transform(PCs)
        EOS_reconstructed = scaler.inverse_transform(EOS_reconstructed)

        # compute the speed of sound square
        cs2_original = []
        cs2_reconstructed = []
        for i in range(number_of_EoS):
            cs2_original.append(compute_speed_of_sound_square(T_plot, EOS_set[i]))
            cs2_reconstructed.append(compute_speed_of_sound_square(T_plot, EOS_reconstructed[i]))
        cs2_original = np.array(cs2_original)
        cs2_reconstructed = np.array(cs2_reconstructed)

        RMS_error = np.sqrt(
            np.mean((cs2_original - cs2_reconstructed)**2, axis=0))
        plt.plot(T_plot, RMS_error,
                 label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
    plt.legend()
    plt.xlim([T_plot[0], T_plot[-1]])
    #plt.ylim([0, None])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$c_s^2$ RMS error")
    plt.yscale('log')
    plt.savefig("RMS_errors_EOS_cs2.png")
    plt.clf()

    # print out the PCs ranges
    pca = PCATransformation(0.5)
    PCs = pca.fit_transform(EOS_set)
    print(f"Number of components = {PCs.shape[1]}")
    print(PCs.min(axis=0), PCs.max(axis=0))
    for i in range(PCs.shape[1]):
        plt.figure()
        plt.hist(PCs[:, i], bins=17, density=True)
        plt.savefig(f"PC{i}.png")
        plt.clf()

    with open("EoSPCA.pickle", "wb") as f:
        pickle.dump(pca, f)

    with open("EoSPCAChain.pickle", "wb") as f:
        pickle.dump(PCs, f)

    # Compute the 90% prior for the shear viscosity from the eta_s_set for each T
    hotQCD_T = validation_data[::4, 0]
    hotQCD_PoverT4 = validation_data[::4, 1]
    EOS_set = np.array(EOS_set)
    low, high = np.percentile(EOS_set, [0, 100.], axis=0)
    plt.fill_between(T_plot, low/T_plot**4, high/T_plot**4, alpha=0.2, color='b', label='100% prior')
    plt.plot(T_plot, high/T_plot**4, 'b-', lw=1)
    plt.plot(T_plot, low/T_plot**4, 'b-', lw=1)
    low, high = np.percentile(EOS_set, [50-90/2., 50+90/2.], axis=0)
    plt.fill_between(T_plot, low/T_plot**4, high/T_plot**4, alpha=0.2, color='g', label='90% prior')
    plt.plot(T_plot, high/T_plot**4, 'g-', lw=1)
    plt.plot(T_plot, low/T_plot**4, 'g-', lw=1)
    low, high = np.percentile(EOS_set, [50-50/2., 50+50/2.], axis=0)
    plt.fill_between(T_plot, low/T_plot**4, high/T_plot**4, alpha=0.2, color='r', label='50% prior')
    plt.plot(T_plot, high/T_plot**4, 'r-', lw=1)
    plt.plot(T_plot, low/T_plot**4, 'r-', lw=1)

    plt.plot(hotQCD_T, hotQCD_PoverT4, 'k-')
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$P/T^4$")
    plt.xlim([0.05, 0.6])
    plt.ylim([0, None])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"EOS_prior.png")
    plt.clf()

    # Compute the 90% prior for the speed of sound square from the EOS_set for each T
    cs2_original = []
    for i in range(number_of_EoS):
        cs2_original.append(compute_speed_of_sound_square(T_plot, EOS_set[i]))
    cs2_original = np.array(cs2_original)
    low, high = np.percentile(cs2_original, [0, 100.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='b', label='100% prior')
    plt.plot(T_plot, high, 'b-', lw=1)
    plt.plot(T_plot, low, 'b-', lw=1)
    low, high = np.percentile(cs2_original, [50-90/2., 50+90/2.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='g', label='90% prior')
    plt.plot(T_plot, high, 'g-', lw=1)
    plt.plot(T_plot, low, 'g-', lw=1)
    low, high = np.percentile(cs2_original, [50-50/2., 50+50/2.], axis=0)
    plt.fill_between(T_plot, low, high, alpha=0.2, color='r', label='50% prior')
    plt.plot(T_plot, high, 'r-', lw=1)
    plt.plot(T_plot, low, 'r-', lw=1)

    hotQCD_cs2 = compute_speed_of_sound_square(hotQCD_T, hotQCD_PoverT4*hotQCD_T**4)
    plt.plot(hotQCD_T, hotQCD_cs2, 'k-')
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$c_s^2$")
    plt.xlim([0.05, 0.6])
    plt.ylim([0, 0.4])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cs2_prior.png")
    plt.clf()


if __name__ == "__main__":
    ranSeed = 23
    number_of_EoS = 2000
    bLogFlag = True
    min_T_mask_region = 0.10
    max_T_mask_region = 0.50
    use_anchor_point = False
    anchor_point = (0.22, 3.) # anchor point for the GP (T, P/T^4)
    main(ranSeed, number_of_EoS, min_T_mask_region, max_T_mask_region, 
         bLogFlag, use_anchor_point, anchor_point)
