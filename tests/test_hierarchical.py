"""Tests for `anamod` package."""

import os
import sys
from unittest.mock import patch

from anamod.core import model_loader
from anamod.simulation import simulation
from tests.utils import pre_test, post_test


# pylint: disable = invalid-name, protected-access
def test_simulation_flat_hierarchy(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with flat hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation"
           " -seed 0 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -analysis_type hierarchical -hierarchy_type flat -cleanup 0"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_simulation_random_hierarchy(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation"
           " -seed 1 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -analysis_type hierarchical -contiguous_node_names 1 -hierarchy_type random -perturbation permutation -cleanup 0"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_simulation_clustering_hierarchy(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation"
           " -seed 2 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -analysis_type hierarchical -contiguous_node_names 1 -hierarchy_type cluster_from_data -perturbation permutation -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_simulation_permutation_perturbation(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation"
           " -seed 3 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -contiguous_node_names 1 -hierarchy_type random -perturbation permutation -cleanup 0"
           f" -analysis_type hierarchical -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)
