"""Tests for temporal model analysis"""

import json
import logging
import os
import sys
from unittest.mock import patch

from anamod import model_loader
from anamod.simulation import simulation, run_trials
from tests.utils import pre_test, write_logfile


# pylint: disable = protected-access, invalid-name
def test_condor_simulation_regressor1(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with regression model over temporal data"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 100 -analysis_type temporal -num_instances 100 -num_features 10"
           " -model_type regressor -num_shuffling_trials 10"
           " -noise_type additive_gaussian -noise_multiplier 0.001"
           " -fraction_relevant_features 0.5 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        summary = simulation.main()
    write_logfile(caplog, output_dir)
    file_regression.check(json.dumps(summary, indent=2), extension=".json")


def test_condor_simulation_classifier1(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with classification model over temporal data and BCE losses computed w.r.t. labels"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 100 -analysis_type temporal -num_instances 100 -num_features 10"
           " -model_type classifier -num_shuffling_trials 10"
           "  -noise_type additive_gaussian -noise_multiplier 0.001"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           " -fraction_relevant_features 0.5 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        summary = simulation.main()
    write_logfile(caplog, output_dir)
    file_regression.check(json.dumps(summary, indent=2), extension=".json")


def test_condor_simulation_classifier2(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with classification model over temporal data and RMSE losses computed w.r.t. predictions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 100 -analysis_type temporal -num_instances 100 -num_features 10"
           " -model_type classifier -num_shuffling_trials 10"
           "  -noise_type additive_gaussian -noise_multiplier 0.001 -loss_target_values baseline_predictions"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           " -fraction_relevant_features 0.5 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        summary = simulation.main()
    write_logfile(caplog, output_dir)
    file_regression.check(json.dumps(summary, indent=2), extension=".json")


def test_condor_trial_regressor1(file_regression, tmpdir, caplog, shared_fs):
    """Test trial with regression model over temporal data"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.run_trials -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -start_seed 1000 -num_trials 2"
           " -type test -wait_period 0 -trial_wait_period 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    strargs = " ".join(cmd.split()[3:])
    summary = run_trials.main(strargs)
    write_logfile(caplog, output_dir)
    file_regression.check(json.dumps(summary, indent=2), extension=".json")
