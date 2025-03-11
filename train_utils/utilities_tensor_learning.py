import numpy as np
import itertools
import tensap

from joblib import Parallel, delayed


def is_consecutive(numbers):
    return all(numbers[i] == numbers[i - 1] + 1 for i in range(1, len(numbers)))


def absolute_error(true, pred):
    return np.linalg.norm(true - pred, ord=2)


def extract_non_consecutive(a):
    a = np.array(a)
    mask = np.diff(a) != 1
    # Find the index where non-consecutive elements start
    idx = np.where(mask)[0]
    # If there are no non-consecutive elements, return an empty list
    if len(idx) == 0:
        return []
    # Return the non-consecutive elements from the first non-consecutive index
    return a[idx[0] + 1:].tolist()


def relative_error(S, S_approx):
    return np.linalg.norm(S - S_approx, 'fro') / np.linalg.norm(S, 'fro')


class CPN_LR(object):
    def __init__(self, S, Sref, tol_eps, alpha=1., beta=1 / np.sqrt(2)):
        super(CPN_LR, self).__init__()
        self.tol_eps = float(tol_eps)
        self.S = S
        self.Sref = Sref
        self.alpha = alpha
        self.beta = beta
        assert self.tol_eps < 1

    def truncate_svd(self, U):
        ## Start at the end
        Q = U.T @ (self.S - self.Sref)
        r = Q.shape[0] - 1
        tol_check = (np.linalg.norm(self.S, 'fro') * (self.tol_eps * self.beta)) ** 2
        Q_norm = np.linalg.norm(Q[r, :], ord=2) ** 2
        while Q_norm <= tol_check:
            r -= 1
            Q_norm += np.linalg.norm(Q[r, :], ord=2) ** 2
        return U[:, :r + 1], Q[:r + 1, :]

    def weights(self, index, indices_list, learnt_weights):
        X = 1 - np.sum(learnt_weights)
        w = (index ** (self.alpha) / np.sum(np.array(indices_list) ** (self.alpha))) * X
        return w

    def tol_eps_wise(self, index, indices_list, learnt_weights):

        return (self.tol_eps * np.sqrt(self.weights(index, indices_list,
                                                    learnt_weights)) * np.linalg.norm(self.S, 'fro')) * np.sqrt(
            1 - self.beta ** 2)

    @staticmethod
    def norm_i(b, index_n, gamma_list):

        if len(b.shape) == 1:
            return np.max(np.linalg.norm(b[index_n], ord=2), (1 / gamma_list) * np.abs(np.delete(b, index_n, axis=0)))
        elif len(b.shape) == 2:
            if len(gamma_list) == 0:
                return np.linalg.norm(b[index_n], ord=2, axis=0)
            else:
                return np.maximum(np.linalg.norm(b[index_n], ord=2, axis=0),
                                  np.max((1 / gamma_list)[:, np.newaxis] * np.abs(np.delete(b, index_n, axis=0)),
                                         axis=0))
        elif len(b.shape) == 3:
            if len(gamma_list) == 0:
                return np.linalg.norm(b[index_n], ord=2, axis=0)
            else:
                return np.maximum(np.linalg.norm(b[index_n], ord=2, axis=0), np.max(
                    (1 / gamma_list)[:, np.newaxis, np.newaxis] * np.abs(np.delete(b, index_n, axis=0)), axis=0))

    def lip_norm_i(self, X, Y, index_n, gamma_list):

        # diff = X[:, :, np.newaxis] - X[:, np.newaxis, :]
        diff_1 = X[index_n, :, np.newaxis] - X[index_n, np.newaxis, :]
        diff_2 = np.delete(X[:, :, np.newaxis], index_n, axis=0) - np.delete(X[:, np.newaxis, :], index_n, axis=0)
        diff = np.concatenate((diff_1, diff_2), axis=0)
        norm_diff = self.norm_i(diff, index_n, gamma_list)
        norm_diff = np.where(norm_diff == 0, np.inf, norm_diff)
        closest_neighbor = np.argmin(norm_diff, axis=1)
        X_close = X[:, closest_neighbor]
        Y_close = Y[closest_neighbor]

        ratio = np.abs(Y - Y_close) / self.norm_i(X - X_close, index_n, gamma_list)
        return np.max(ratio)

    @staticmethod
    def lip_norm_2(X, Y):

        diff = X[:, :, np.newaxis] - X[:, np.newaxis, :]
        norm_diff = np.linalg.norm(diff, ord=2, axis=0)  ####norme i
        norm_diff = np.where(norm_diff == 0, np.inf, norm_diff)
        closest_neighbor = np.argmin(norm_diff, axis=1)
        X_close = X[:, closest_neighbor]
        Y_close = Y[closest_neighbor]

        ratio = np.abs(Y - Y_close) / np.linalg.norm(X - X_close, axis=0)
        return np.max(ratio)

    def gamma_i(self, index, indices_list, learnt_weights, Gamma):
        gamma = np.sqrt(self.weights(index, indices_list, learnt_weights) * (Gamma ** 2 - 1))
        return gamma

    def n_min(self, U, Q, tol_min):
        n = 1
        S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        err = tol_min * np.linalg.norm(self.S, 'fro')
        while np.linalg.norm(self.S - S_approx, 'fro') > err:
            n += 1
            S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        return n

    def tensor_solver(self, SOLVER, input_data, output_data, ntrain):

        SOLVER.training_data = [None, output_data[:ntrain]]
        SOLVER.test_data = [input_data, output_data]

        F, _ = SOLVER.solve()
        return F

    def find_n(self, U, Q, p1, tol_min=1., train_set=1., Gamma=100, basisAdaptation=False, poly_space=None):
        coeffs_dict = {}
        func = {}
        N_train = int(Q.shape[1] * train_set)
        for j, q_j in enumerate(Q):
            coeff_name = f"coef{j}"
            coeffs_dict[coeff_name] = {'value': q_j, 'index': j}
        n = self.n_min(U, Q, float(tol_min))
        print("n min     =   ", n)
        index_r = list(np.arange(n))
        for k in index_r:
            coeffs_dict.pop(f"coef{k}")
        indices_list = [value["index"] for value in coeffs_dict.values()]
        dim = n

        Qr = Q[:n, :]
        Q_check = Q[:n, :]

        learnt_weights = []
        lipschitz_consts = []
        lip_consts_inputs = []
        while indices_list:  # step
            learnt_w = []
            learnt_g = []
            lip_consts = []
            print(f"#################################################################step{dim}")

            X = [tensap.UniformRandomVariable(np.min(x), np.max(x)) for x in Q_check]
            BASIS = [
                tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(p1 + 1))
                for x in X
            ]
            BASES = tensap.FunctionalBases(BASIS)
            SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
                dim, tensap.SquareLossFunction()
            )

            SOLVER.bases = BASES
            SOLVER.bases_eval = BASES.eval(Q_check.T[:N_train, :])

            SOLVER.tolerance["on_stagnation"] = 1e-6

            SOLVER.initialization_type = "canonical"

            SOLVER.linear_model_learning.regularization = True
            # SOLVER.linear_model_learning.regularization_type = "l1"
            SOLVER.linear_model_learning.basis_adaptation = True
            SOLVER.linear_model_learning.error_estimation = True

            SOLVER.test_error = True
            # SOLVER.bases_eval_test = BASES.eval(X_TEST)

            SOLVER.rank_adaptation = True
            SOLVER.rank_adaptation_options["max_iterations"] = 20
            SOLVER.rank_adaptation_options["theta"] = 0.8
            SOLVER.rank_adaptation_options["early_stopping"] = True
            SOLVER.rank_adaptation_options["early_stopping_factor"] = 10

            if dim == 2:
                SOLVER.tree_adaptation = False
            else:
                SOLVER.tree_adaptation = True
            SOLVER.tree_adaptation_options["max_iterations"] = 1e2
            # SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

            SOLVER.alternating_minimization_parameters["stagnation"] = 1e-10
            SOLVER.alternating_minimization_parameters["max_iterations"] = 50

            SOLVER.display = False
            SOLVER.alternating_minimization_parameters["display"] = False

            SOLVER.model_selection = True
            SOLVER.model_selection_options["type"] = "test_error"

            coeffs = Q[indices_list, :].T
            SOLVER.tolerance["on_error"] = self.tol_eps * 0.1

            f = Parallel(n_jobs=-1)(
                delayed(self.tensor_solver)(SOLVER, Q_check.T, coeffs[:, i], N_train) for i in range(coeffs.shape[1]))

            pred = np.array([f[i](Q_check.T) for i in range(len(f))]).T

            for (i, j) in zip(indices_list, range(len(indices_list))):
                coeff_name = f'coef{i}'
                # g_i = self.gamma_i(i, indices_list, learnt_gamma_i, Gamma)
                g_i = self.gamma_i(i, indices_list, learnt_weights, Gamma)
                lip_const_i = self.lip_norm_i(Q_check, pred[:, j], index_r,
                                              np.array(lip_consts_inputs))

                if absolute_error(coeffs[:, j], pred[:, j]) <= self.tol_eps_wise(i, indices_list,
                                                                                 learnt_weights) and lip_const_i <= g_i:  # and lip_const_i <= g_i
                    func[coeff_name] = {'function': f[j], 'index': i, 'nb_deps': len(Q_check),
                                        'lip_constant': lip_const_i}
                    deleted_coeff = coeffs_dict.pop(coeff_name)
                    print("[", min(np.arange(dim)) + 1, "...", max(np.arange(dim)) + 1, "]", '|->',
                          deleted_coeff['index'] + 1)
                    w = self.weights(i, indices_list, learnt_weights)
                    learnt_w.append(w)
                    learnt_g.append(g_i)
                    lip_consts.append(lip_const_i)

            learnt_weights.extend(learnt_w)
            next_coef = f'coef{dim}'
            if next_coef in coeffs_dict:
                q_next = coeffs_dict.pop(next_coef)
                Qr = np.concatenate([Qr, q_next['value'][None,]], axis=0)
                index_r.append(q_next['index'])
                Q_check = np.concatenate([Q_check, q_next['value'][None,]], axis=0)
            else:
                nb_deps = func[next_coef]['nb_deps']
                q_next_approx = func[next_coef]['function'](Q_check.T[:, :nb_deps])
                Q_check = np.concatenate([Q_check, q_next_approx[None,]], axis=0)
                # gamma_list.append(self.gamma_i(func[next_coef]['index'], indices_list, learnt_gamma_i, Gamma))
                lip_consts_inputs.append(func[next_coef]['lip_constant'])
            lipschitz_consts.extend(lip_consts)
            # learnt_gamma_i.extend(learnt_g)
            indices_list = [value['index'] for value in coeffs_dict.values()]
            dim += 1

            print("Rest to learn = ", len(indices_list))

        print('Done !\t Dimension of the manifold = ', len(Qr))
        print('Index r = ', [r + 1 for r in index_r])

        return Qr, dict(sorted(func.items(), key=lambda x: x[1]["index"])), index_r, lipschitz_consts

    def coeff_approximation(self, Q, func, index_r):
        Qbar = np.zeros((len(func.items()), Q.shape[1]))
        list_keys = list(func.keys())
        Q_total = np.zeros((len(index_r) + len(func.items()), Q.shape[1]))
        Q_total[index_r, :] = Q
        for i, coef_name in enumerate(list_keys):
            nb_deps = func[coef_name]['nb_deps']
            Qbar[i, :] = func[coef_name]['function'](Q_total.T[:, :nb_deps])
            Q_total[func[coef_name]['index']] = Qbar[i, :]
        return Qbar
