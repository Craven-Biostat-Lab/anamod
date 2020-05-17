"""Feature class"""
import anytree
import numpy as np
import xxhash

from anamod import constants


# pylint: disable = too-many-instance-attributes
class Feature(anytree.Node):
    """Class representing feature/feature group"""
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.parent_name = kwargs.get(constants.PARENT_NAME, "")
        self.description = kwargs.get(constants.DESCRIPTION, "")
        self.idx = kwargs.get("idx", [])
        self.perturbable = kwargs.get("perturbable", True)
        # TODO: (Verify) Could initialize the RNG right away, since cloudpickle should stick be able to pickle it
        self._rng_seed = xxhash.xxh32_intdigest(name)
        self.rng = None  # RNG used for shuffling this feature - see perturbations.py: 'feature.rng'
        # Importance attributes
        self.important = False
        self.temporally_important = False
        self.temporal_window = None
        self.window_important = False

    @property
    def rng_seed(self):
        """Get RNG seed"""
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, seed):
        """Set RNG seed"""
        self._rng_seed = seed

    def initialize_rng(self):
        """Initialize random number generator for feature (used for shuffling perturbations)"""
        self.rng = np.random.default_rng(self._rng_seed)

    def uniquify(self, uniquifier):
        """Add uniquifying identifier to name"""
        assert uniquifier
        self.name = "{0}->{1}".format(uniquifier, self.name)

    @property
    def size(self):
        """Return size"""
        return len(self.idx)
