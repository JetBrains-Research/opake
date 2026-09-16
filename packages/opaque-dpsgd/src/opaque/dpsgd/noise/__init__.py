"""DP-SGD noise mechanisms façade — Gaussian and projected JME.

States and JME allocation types live in :mod:`opaque.dpsgd.noise.types`.
"""

from opaque.api.dpsgd.noise import gaussian_noise, jme_noise

__all__ = ["gaussian_noise", "jme_noise"]
