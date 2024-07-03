import numpy as np
import scipy
import scipy.sparse as sp

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import (
    multiexpm,
    multieye,
    multiqr,
    multisym,
    multitransp,
)

class Generalised_Stiefel(Manifold):
    r"""
    TODO: Update docstring for Generalised case
    TODO: Matmul using sparse matrices for k>1
    The (product) Stiefel manifold.

    The Stiefel manifold :math:`\St(n, p)` is the manifold of orthonormal ``n x
    p`` matrices.
    A point :math:`\vmX \in \St(n, p)` therefore satisfies the condition
    :math:`\transp{\vmX}\vmX = \Id_p`.
    Points on the manifold are represented as arrays of shape ``(n, p)`` if
    ``k == 1``.
    For ``k > 1``, the class represents the product manifold of ``k`` Stiefel
    manifolds, in which case points on the manifold are represented as arrays
    of shape ``(k, n, p)``.

    The metric is the usual Euclidean metric on :math:`\R^{n \times p}` which
    turns :math:`\St(n, p)^k` into a Riemannian submanifold.

    Args:
        n: The number of rows.
        p: The number of columns.
        k: The number of elements in the product.
        B: Matrix
        Binv: Inverse matrix to B if known
        retraction: The type of retraction to use.
            Possible choices are ``qr`` and ``polar``.

    Note:
        The formula for the exponential map can be found in [ZH2021]_.

        The Weingarten map is taken from [AMT2013]_.

        The default retraction used here is a first-order one based on
        the QR decomposition.
        To switch to a second-order polar retraction, use ``Stiefel(n, p, k=k,
        retraction="polar")``.
    """

    def __init__(self, n: int, p: int, B, *, Binv = None, k: int = 1, retraction: str = "qr"):
        self._n = n
        self._p = p
        self._k = k
        self.B  = B
        self.Binv = Binv

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )
        if k < 1:
            raise ValueError(f"Need k >= 1. Value supplied was k = {k}")

        if k == 1:
            name = f"Stiefel manifold St({n},{p})"
        elif k >= 2:
            name = f"Product Stiefel manifold St({n},{p})^{k}"
        dimension = int(k * (n * p - p * (p + 1) / 2))
        super().__init__(name, dimension)

        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    @property
    def typical_dist(self):
        return np.sqrt(self._p * self._k)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, self.B@tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return vector - point @ multisym(multitransp(point) @ self.B @ vector)

    to_tangent_space = projection

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        raise NotImplementedError

    def _retraction_polar(self, point, tangent_vector):
        Y = point + tangent_vector
        return self.guf(Y)

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def random_point(self):
        if self._k == 1:
            point = self.guf(np.random.normal(size=(self._n, self._p)))
        else:
            point = self.guf(np.random.normal(size=(self._k, self._n, self._p)))
        return point

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape)
        vector = self.projection(point, vector)
        return vector / np.linalg.norm(vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)
    
    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        if self.Binv is not None:
            egrad_scaled = self.Binv@euclidean_gradient
        else:
            egrad_scaled = scipy.linalg.solve(self.B,euclidean_gradient)
        rgrad = egrad_scaled - point@multisym(multitransp(point)@euclidean_gradient)
        return rgrad

    def zero_vector(self, point):
        if self._k == 1:
            return np.zeros((self._n, self._p))
        return np.zeros((self._k, self._n, self._p))
    
    def guf(self,Y):
        U, _, Vh = np.linalg.svd(Y, full_matrices=False)
        ssquare, q = np.linalg.eig(multitransp(U)@(self.B@U))
        qsinv = np.einsum('...nm,...m->...nm',q,1/np.sqrt(ssquare))
        X = U@((qsinv@multitransp(q))@Vh)
        return X.real