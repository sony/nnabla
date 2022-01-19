# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce

import numpy as np
from scipy.linalg import pinv


class ALS(object):
    """Alternating Least Square to solve Canonical Polyadic Decompositon.

    """

    def __init__(self, ):
        # self.pool = ThreadPool(mp.cpu_count() - 1)
        pass

    def solve(self, X, rank, max_iter=500, stopping_criterion=1e-5,
              lambda_reg=0.0, dtype=np.float32, rng=None):
        """Solve CPD

        Args:
            X (numpy.ndarray): Target tensor of CPD.
            rank (int): Rank of the approximate tensor.
            max_iter (int): Max iteration of the ALS.
            stopping_criterion (float): Threshold for stopping the ALS.
                If the value is negative, the convergence check is ignored;
                in other words, it may reduce the computation time.
            lambda_reg (float): regularization parameter. Larger lambda_reg
                means larger regularization.
            dtype (numpy.dtype): Data type

        Returns:
            list of numpy.ndarray: Decomposed matrices.
            numpy.ndarray: Lambda of the CPD.

        """

        N = X.ndim  # Tensor dimensions
        squared_norm_X = np.sum(X ** 2)  # Frobenious norm square

        # Initialize
        if rng is None:
            rng = np.random.RandomState(313)
        A = [None for _ in range(N)]
        for n in range(1, N):
            A[n] = np.array(rng.rand(X.shape[n], rank), dtype=dtype)

        # Solve ALS problem
        criterion = 0
        for itr in range(max_iter):
            criterion_prev = criterion

            # Fix one dimension
            for n in range(N):
                # Solve sub problem
                V = self.hadamard_products_of_gramians(A, n)
                P = self.khatrirao_products(A, n)
                A_n = np.tensordot(
                    X,
                    P,
                    axes=(
                        [o for o in range(N) if o != n],
                        np.arange(N-1)[::-1]))
                # L2 regularization
                V = V+np.eye(rank)*lambda_reg

                A_n = A_n.dot(pinv(V))

                # Normalize
                if itr == 0:
                    lmbda = np.sqrt((A_n ** 2).sum(axis=0))
                else:
                    lmbda = A_n.max(axis=0)
                    lmbda[lmbda < 1] = 1
                A[n] = A_n / lmbda

            # Check convergence
            if stopping_criterion < 0:
                continue
            X_approx = self.approximate_tensor(A, lmbda)
            squared_norm_residual = squared_norm_X + \
                np.sum(X_approx**2) - 2 * np.sum(X*X_approx)
            criterion = 1.0 - (squared_norm_residual / squared_norm_X)
            criterion_change = abs(criterion_prev - criterion)

            if itr > 0 and criterion_change < stopping_criterion:
                break

        return A, lmbda

    def khatrirao_product(self, X, Y):
        """Column-wise Khatri-Rao product.

        Khatri-Rao product is the column-wise matching Kroncker product.
        For examples, suppose matrices X and Y each of which dimensions
        is (I, K) and (J, K).

        Khatri-Rao product between X and Y is defined by

        .. math::

            X \odot Y = [a_1 \otimes b_1   a_2 \otimes b_2  \ldots a_K \otimes b_K ].


        Khatri-Rao product usually returns a matrix, or 2-dimensional array,
        but in this function, it returns 3-dimensional array for the next use for
        tensor-dot product efficiently.

        Args:
            X (numpy.ndarray): the matrix which the number of the columns is the same as Y's.
            Y (numpy.ndarray): the matrix which the number of the columns is the same as X's.

        Returns:
            numpy.ndarray: the 3-dimensional array with the shape (I x J x K).

        """

        n_cols = X.shape[1]
        P = np.zeros((X.shape[0], Y.shape[0], n_cols))
        for r in range(n_cols):
            outer_product = np.outer(X[:, r], Y[:, r])  # TODO: dimension
            P[:, :, r] = outer_product

#         # Multi-thread
#         def outer_prod(X, Y, r, P):
#             P[:, :, r] = np.outer(X[:, r], Y[:, r])
#             return None
#         future_list = []
#         for r in range(n_cols):
#             outer_product = np.outer(X[:, r], Y[:, r])  #TODO: dimension
#             P[:, :, r] = outer_product
#             future = self.pool.apply_async(outer_prod, (X, Y, r, P))
#             future_list.append(future)
#         for r in range(n_cols):
#             future_list[r].get()

        return P

    def khatrirao_products(self, A, n, reverse=True):
        """Sequential Khatri-Rao product without the `n`-th matrix in the reverse order.

        Args:
            A (list of numpy.ndarray): matrices where the number of columns for each matrix is the same.
            n (int): N-th matrix which is omitted when computing Khatri-Rao product.

        Returns:
            numpy.ndarray: Result of the reduction of Khatri-Rao product for `A`.
        """
        order = list(range(n)) + list(range(n + 1, len(A)))
        order = sorted(order, reverse=reverse)
        Z = reduce(lambda X, Y: self.khatrirao_product(
            X, Y), [A[o] for o in order])
        return Z

    def hadamard_products_of_gramians(self, A, n):
        """Hadamard productsof the gramian matrix without the `n`-th matrix.

        Args:
            A (list of numpy.ndarray): matrices where the number of columns for each matrix is the same.
            n (int): N-th matrix which is omitted when computing Gramian matrix.

        """
        order = list(range(n)) + list(range(n + 1, len(A)))
        V = 1.
        for o in order:
            V = V * np.dot(A[o].T, A[o])
        return V

    def approximate_tensor(self, A, lmbda):
        """Compute the approximate original tensor

        Args:
            A (list of numpy.ndarray): matrices where the number of columns for each matrix is the same.
            lmbda (numpy.ndarray): 1-dimensional np.ndarray
        """
        rank = len(lmbda)
        Z = np.zeros([a.shape[0] for a in A])
        for r in range(rank):
            a0 = A[0][:, r]
            for a in A[1:]:
                a0 = np.multiply.outer(a0, a[:, r])
            Z[...] += lmbda[r] * a0

#         # Multi-thread (takes much memory)
#         def outer_prod(A, lmbda, r):
#             a0 = A[0][:, r]
#             for a in A[1:]:
#                 a0 = np.multiply.outer(a0, a[:, r])
#             return lmbda[r] * a0
#         future_list = []
#         for r in range(rank):
#             future = self.pool.apply_async(outer_prod, (A, lmbda, r))
#             future_list.append(future)
#         results = []
#         for r in range(rank):
#             results.append(future_list[r].get())
#         Z = reduce(lambda X, Y: X + Y, results)

        return Z
