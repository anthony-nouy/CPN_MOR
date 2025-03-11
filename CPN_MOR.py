import matplotlib.pyplot as plt
from train_utils.utilities_tensor_learning import CPN_LR
from train_utils.utilities_sparse import *

from train_utils.data_loader import myloader
from pathlib import Path
from timeit import default_timer
import yaml
from argparse import ArgumentParser
from visualization.tree_viz import tree_visualization


def error_svd(singular_values):
    error = np.sqrt(np.flip(np.cumsum(np.flip(singular_values ** 2)) / (singular_values ** 2).sum()))
    return error


def run():
    folder = Path(results_path)
    folder.mkdir(parents=True, exist_ok=True)
    path_left_rob = config["path_svd"]
    path_rob = Path(path_left_rob)
    if path_rob.exists() and recompute_svd == False:
        U = np.load(path_left_rob)
    else:
        print("Computing SVD...")
        U, Sigma, _ = np.linalg.svd(S - Sref, full_matrices=False)
        plt.semilogy(range(1, len(error_svd(Sigma))), error_svd(Sigma)[1:], marker='o')
        plt.grid()
        plt.savefig(results_path+"/singular_value_decay.png")
    np.save(results_path + "/left_rob.npy", U)
    print("SVD truncation...")

    Vstar, Qstar = method.truncate_svd(U[:, ])

    print("Vstar shape= ", Vstar.shape)
    print("Qstar shape= ", Qstar.shape)

    t1 = default_timer()

    Gamma = config["add_params"]["L"]
    train_set = config["params"]["train_val_set"]
    tol_min = config["add_params"]["tol_min"]

    Qr, func, index_r, lipschitz_consts = method.find_n(Vstar, Qstar, p1=p1, tol_min=tol_min,
                                      train_set=train_set, Gamma=Gamma)

    np.save(results_path + "/Qr.npy", Qr)
    np.save(results_path + "/Vstar.npy", Vstar)
    np.save(results_path + "/Qstar.npy", Qstar)
    np.save(results_path + "/function.npy", func)
    np.save(results_path + "/index_r.npy", index_r)

    print("Qr shape = ", Qr.shape)
    n = len(index_r)
    V = Vstar[:, index_r]
    Vbar = np.delete(Vstar, index_r, axis=1)
    Qbar = method.coeff_approximation(Qr, func, index_r)
    Qtest = Vstar.T @ (S_test - Sref_test)
    S_lin = Sref + Vstar[:, :n] @ Qstar[:n, :]
    S_lin_test = Sref_test + Vstar[:, :n] @ Qtest[:n, :]
    S_approx = Sref + V @ Qr + Vbar @ Qbar
    t2 = default_timer()

    Qr_test = V.T @ (S_test - Sref_test)
    Qr_pred_test = Qr_test + tol

    Qbar_test = method.coeff_approximation(Qr_test, func, index_r)
    Qbar_online_test = method.coeff_approximation(Qr_pred_test, func, index_r)

    S_approx_test = Sref_test + V @ Qr_test + Vbar @ Qbar_test
    S_approx_test_online = Sref_test + V @ Qr_pred_test + Vbar @ Qbar_online_test

    np.save(results_path + "/Qbar_pred_test.npy", Qbar_test)
    np.save(results_path + "/classical_POD_approach", S_lin_test)
    np.save(results_path + "/our_test_approx.npy", S_approx_test)

    print("time = ", t2 - t1, " secs")
    print("Decoder lipschitz const = ", np.sqrt(1 + sum(np.array(lipschitz_consts) ** 2)))
    print("Linear reconstruction error training = ", relative_error(S, S_lin))
    print("Linear reconstruction error test = ", relative_error(S_test, S_lin_test))
    print("Nonlinear reconstruction error training = ", relative_error(S, S_approx))
    print("Test reconstruction error = ", relative_error(S_test, S_approx_test))
    print("Test reconstruction error online = ", relative_error(S_test, S_approx_test_online))
    tree_visualization(config)

    with open(Path(folder, "train_info.txt"), "w") as f:
        s = ""
        s += f"  * p          = {config['params']['p']:_}\n"
        s += f"  * tolerance          = {config['params']['tolerance']}\n"
        s += f"  * alpha          = {config['add_params']['alpha']:}\n"
        s += f"  * beta          = {config['add_params']['beta']:}\n"
        s += f"  * L          = {config['add_params']['L']:}\n"
        s += f"  * tensor learning          = {config['params']['approximation_type']:}\n"
        f.write(s)

    with open(Path(folder, "results_info.txt"), "w") as f:
        s = ""
        s += f"  * Time          = {t2 - t1:_}\n"
        s += f"  * n          = {len(index_r):_}\n"
        s += f"  * I          = {str(index_r)}\n"
        s += f"  * Linear reconstruction error training          = {relative_error(S, S_lin):}\n"
        s += f"  * Linear reconstruction error test          = {relative_error(S_test, S_lin_test):}\n"
        s += f"  * Nonlinear reconstruction error training          = {relative_error(S, S_approx):}\n"
        s += f"  * Test reconstruction error          = {relative_error(S_test, S_approx_test):}\n"
        s += f"  * Test reconstruction error online          = {relative_error(S_test, S_approx_test_online):}\n"
        s += f"  * Decoder lipschitz constant          = {np.sqrt(1 + sum(np.array(lipschitz_consts) ** 2)):}\n"
        f.write(s)


def test():
    folder = Path(results_path)
    if folder.exists():
        Vstar = np.load(results_path + "/Vstar.npy")
        # Qstar = np.load(results_path + "/Qstar.npy")
        # Qr = np.load(results_path + "/Qr.npy")
        func = np.load(results_path + "/function.npy", allow_pickle=True).item()
        index_r = np.load(results_path + "/index_r.npy")

        n = len(index_r)
        V = Vstar[:, index_r]
        Qr = V.T @ (S - Sref)
        Vbar = np.delete(Vstar, index_r, axis=1)

        Qbar = method.coeff_approximation(Qr, func, index_r)
        Qstar = Vstar.T @ (S - Sref)
        Qstar_test = Vstar.T @ (S_test - Sref_test)
        S_lin = Sref + Vstar[:, :n] @ Qstar[:n, :]
        S_lin_test = Sref_test + Vstar[:, :n] @ Qstar_test[:n, :]
        S_approx = Sref + V @ Qr + Vbar @ Qbar

        Qr_test = V.T @ (S_test - Sref_test)
        Qr_pred_test = Qr_test + tol

        Qbar_test = method.coeff_approximation(Qr_test, func, index_r)
        Qbar_online_test = method.coeff_approximation(Qr_pred_test, func, index_r)

        S_approx_test = Sref_test + V @ Qr_test + Vbar @ Qbar_test
        S_approx_test_online = Sref_test + V @ Qr_pred_test + Vbar @ Qbar_online_test

        print("Linear reconstruction error training = ", relative_error(S, S_lin))
        print("Linear reconstruction error test = ", relative_error(S_test, S_lin_test))
        print("Nonlinear reconstruction error training = ", relative_error(S, S_approx))
        print("Test reconstruction error = ", relative_error(S_test, S_approx_test))
        print("Test reconstruction error online = ", relative_error(S_test, S_approx_test_online))
    else:
        raise ValueError("Results have not been saved yet...Train the model first ! ")

    np.save(results_path + "/Qbar_pred_test.npy", Qbar_test)
    np.save(results_path + "/our_test_approx.npy", S_approx_test)


def plot(config):
    tree_visualization(config)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', default="train", type=str, help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    tol = float(config["params"]["tolerance"])
    alpha = config["add_params"]["alpha"]
    beta = config["add_params"]["beta"]
    p1 = config["params"]["p"]
    recompute_svd = config["add_params"]["compute_svd"]
    results_path = config["path_results"]
    approx_type = config["params"]["approximation_type"]

    S, S_test, Sref, Sref_test = myloader(config)
    if approx_type == "sparse":
        method = CPN_S(S, Sref, tol, alpha=alpha, beta=beta)
    elif approx_type == "low_rank":
        method = CPN_LR(S, Sref, tol, alpha=alpha, beta=beta)
    else:
        raise ValueError("Approximation type not implemented !")

    if args.mode == "train":
        run()
    elif args.mode == "test":
        test()
    else:
        plot(config)
