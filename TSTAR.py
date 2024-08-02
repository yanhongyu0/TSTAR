

import copy

import tensorly as tl
import scipy as sp
import numpy as np
from .util.functions import  fit_ar


class TSTAR(object):
    def __init__(self,ts, p, Rs, K, tol, seed=None, Us_mode=4, \
        verbose=0, convergence_loss=False):
        self._ts = ts
        self._ts_ori_shape = ts.shape
        self._N = len(ts.shape) - 1
        self.T = ts.shape[-1]
        self._p = p
        self._Rs = Rs
        self._K = K
        self._tol = tol
        self._Us_mode = Us_mode
        self._verbose = verbose
        self._convergence_loss = convergence_loss
        if seed is not None:
            np.random.seed()
    def _initilizer(self, T_hat, Js, Rs):
        U = [ np.random.random([j,r]) for j,r in zip( list(Js), Rs )]
        return U

    def _get_cores(self, Xs, Us):
        s=[u.T for u in Us]
        cores = [ tl.tenalg.multi_mode_dot( x, s, modes=[i for i in range(len(Us))] ) for x in Xs]
        return cores

    def _estimate_ar(self, cores, p):
        cores = copy.deepcopy(cores)
        alpha = fit_ar(cores, p)
        return alpha

    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor,list):
            return [ tl.base.fold(ten, mode, shape) for ten in tensor ]
        elif isinstance(tensor, np.ndarray):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _get_unfold_tensor(self, tensor, mode):

        if isinstance(tensor, list):
            return [ tl.base.unfold(ten, mode) for ten in tensor]
        elif isinstance(tensor, np.ndarray):
            return tl.base.unfold(tensor, mode)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _update_Us(self, Us, Xs, unfold_cores, n):

        T_hat = len(Xs)
        M = len(Us)
        begin_idx = self._p

        H = self._get_H(Us, n)
        if self._Us_mode == 1:
            if n<M-1:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
            else:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
        elif self._Us_mode == 2:
            if n<M-1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        elif self._Us_mode == 3:
            As = []
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            a = sp.linalg.pinv(np.sum(As, axis=0))
            b = np.sum(Bs, axis=0)
            temp = np.dot(a, b)
            Us[n] = temp / np.linalg.norm(temp)
        elif self._Us_mode == 4:
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            b = np.sum(Bs, axis=0)
            U_, _, V_ = np.linalg.svd(b, full_matrices=False)
            Us[n] = np.dot(U_, V_)
        elif self._Us_mode == 5:
            if n==0:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        elif self._Us_mode == 6:
            if n==1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        return Us

    def _update_Es(self, es, alpha, beta, unfold_cores, i, n):

        T_hat = len(unfold_cores)
        begin_idx = self._p

        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii+1)] for ii in range(self._p)] , axis=0)
            As.append(unfold_cores[t] - a)
        E = np.sum(As, axis=0)
        for t in range(len(es)):
            es[t][i] = self._get_fold_tensor(E / (2*(begin_idx - T_hat) * beta[i]), n, es[t][i].shape)
        return es

    def _compute_convergence(self, new_U, old_U):

        new_old = [ n-o for n, o in zip(new_U, old_U)]

        a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_U], axis=0)
        return a/b

    def _update_cores(self, n, Us, Xs,  cores, alpha, lam=1):

        begin_idx = self._p
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Us, n)
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            a = np.sum([ alpha[i] * self._get_unfold_tensor(cores[t-(i+1)], n) for i in range(self._p)], axis=0 )
            unfold_cores[t] = 1/(1+lam) * (lam * np.dot( np.dot(Us[n].T, unfold_Xs), H.T) + a)
        return unfold_cores

    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs

    def _get_H(self, Us, n):
        ab=Us[::-1]
        Hs = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= n ])
        return Hs

    def run(self):
        result, loss = self._run()

        if self._convergence_loss:

            return result, loss

        return result, None

    def _run(self):
        Xs = self._get_Xs(self._ts)

        con_loss = []

        Us = self._initilizer(len(Xs), Xs[0].shape, self._Rs)
        for k in range(self._K):
            old_Us = Us.copy()

            cores = self._get_cores(Xs, Us)

            alpha = self._estimate_ar(cores, self._p)

            for n in range(len(self._Rs)):
                cores_shape = cores[0].shape
                unfold_cores = self._update_cores(n, Us, Xs, cores, alpha, lam=1)
                cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
                Us = self._update_Us(Us, Xs, unfold_cores, n)

            convergence = self._compute_convergence(Us, old_Us)
            con_loss.append(convergence)

            if k%10 == 0:
                if self._verbose == 1:
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))

            if self._tol > convergence:
                if self._verbose == 1:
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    print("alpha: {}".format(alpha))
                break


        cores = self._get_cores(Xs, Us)
        alpha = self._estimate_ar(cores, self._p)
        new_core = np.sum([al * core for al, core in zip(alpha, cores[-self._p:][::-1])], axis=0)
        new_X = tl.tenalg.multi_mode_dot(new_core, Us)
        Xs.append(new_X)
        return Xs, con_loss
