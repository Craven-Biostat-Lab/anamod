"""Common utility functions"""
import logging
from collections import namedtuple
import contextlib
import os
import sys
import time

import numpy as np
try:
    # TODO: Add note about installing htcondor to documentation
    import htcondor
    from htcondor import JobEventType, JobAction
except ImportError:
    pass  # Caller performs its own check to validate condor availability

from anamod.constants import VIRTUAL_ENV


def get_logger(name, filename, level=logging.INFO):
    """Return logger configure to write to filename"""
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=level, filename=filename, format=formatting)  # if not already configured
    logger = logging.getLogger(name)
    return logger


def round_value(value, decimals=4):
    """Round input to avoid floating-point errors"""
    return np.around(value, decimals=decimals)


Filenames = namedtuple("Filenames", ["exec_filename", "log_filename", "out_filename", "err_filename"])


class CondorJobWrapper():
    """Schedule jobs using condor"""
    # pylint: disable = too-many-instance-attributes
    idx = 0

    def __init__(self, cmd, input_files, job_dir, **kwargs):
        """
        Creates htcondor job wrapper. Use 'run' to submit job and 'monitor' to monitor the status of submitted jobs.
        Args:
        * cmd: command to run on condor execute node
            * If non-shared FS, input file/directory paths supplied to cmd must be stripped of submit node directory structure
            * If shared FS, input file/directory paths supplied to cmd must be absolute paths, since cmd will be run from inside job_dir
       * input_files: list of input files for cmd (must be accessible from submit node)
            * If non-shared FS, these will be transferred to root working directory in execute node
        * job_dir: empty directory for condor logs (submit node) and job outputs (execute node) (will be created if it doesn't exist)
            * If non-shared FS, outputs in this directory will be transferred back to submit node from execute node
        Keyword args:
        * shared_filesystem: Flag to specify shared/non-shared FS
        * memory: amount of memory to request on condor execute node, default 1GB
        * disk: amount of disk storage to request on condor execute node, default 4GB
        * package: software package to install via pip on execute node, default cloudbopper/anamod (relevant for non-shared FS only)
        Other considerations:
        * If non-shared FS, software downloaded and installed in execute node from github package cloudbopper/anamod.git
        * If shared FS, assumes that the submit node code is running inside virtualenv and tries to activate this on execute node
        """
        # Distinguish jobs for monitoring
        self.name = f"job_{CondorJobWrapper.idx}"
        CondorJobWrapper.idx += 1
        # Set up job input files and working directory
        self.cmd = cmd
        self.input_files = ([os.path.abspath(input_file) for input_file in input_files])
        for input_file in self.input_files:
            assert os.path.exists(input_file)
        self.input_files += [f"http://proxy.chtc.wisc.edu/SQUID/chtc/python3{sys.version_info.minor}.tar.gz"]
        self.job_dir = os.path.abspath(job_dir)  # Directory for job logs/outputs in submit host
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)
        self.job_dir_remote = os.path.basename(self.job_dir.rstrip("/"))
        self.filenames = Filenames(exec_filename=f"{self.job_dir}/{self.name}.sh",
                                   log_filename=f"{self.job_dir}/{self.name}.log",
                                   out_filename=f"{self.job_dir}/{self.name}.out",
                                   err_filename=f"{self.job_dir}/{self.name}.err")
        # Process keyword args
        self.shared_filesystem = kwargs.get("shared_filesystem", False)
        memory = kwargs.get("memory", "1GB")
        disk = kwargs.get("disk", "4GB")
        package = kwargs.get("package", "git+https://github.com/cloudbopper/anamod")
        # Create job
        self.job = self.create_job(memory, disk, package)
        self.cluster_id = -1  # set by running job

    def create_job(self, memory, disk, package):
        """Create job"""
        self.create_executable(package)
        # TODO: add options for automatic job retries https://bcg.biostat.wisc.edu/condor-best-practices/
        job = htcondor.Submit({"initialdir": f"{self.job_dir}",
                               "executable": f"{self.filenames.exec_filename}",
                               "output": f"{self.filenames.out_filename}",
                               "error": f"{self.filenames.err_filename}",
                               "log": f"{self.filenames.log_filename}",
                               "request_memory": f"{memory}",
                               "request_disk": f"{disk}",
                               "universe": "vanilla",
                               "should_transfer_files": "NO" if self.shared_filesystem else "YES",
                               "transfer_input_files": "" if self.shared_filesystem else ",".join(self.input_files),
                               "transfer_output_files": "" if self.shared_filesystem else f"{self.job_dir_remote}/",
                               # Send the job to Held state on failure
                               "on_exit_hold": "(ExitBySignal == True) || (ExitCode != 0)",
                               # Periodically retry the jobs every 10 minutes, up to a maximum of 5 retries.
                               "periodic_release": "(NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 600)"
                               })
        return job

    def create_executable(self, package):
        """Create executable shell script"""
        with open(self.filenames.exec_filename, "w") as exec_file:
            # Setup environment and inputs
            exec_file.write("#!/bin/sh\n")
            if not self.shared_filesystem:
                exec_file.write(f"mkdir {self.job_dir_remote}\n"
                                f"tar -xzf python3{sys.version_info.minor}.tar.gz\n"
                                "export PATH=${PWD}/python/bin/:${PATH}\n"
                                "export PYTHONPATH=${PWD}/packages\n"
                                "export LC_ALL=en_US.UTF-8\n"
                                "python3 -m pip install --upgrade pip\n"
                                f"python3 -m pip install {package} --target ${{PWD}}/packages\n")
            else:
                virtualenv = os.environ.get(VIRTUAL_ENV, "")
                exec_file.write(f"source {virtualenv}/bin/activate\n")
            # Execute command
            exec_file.write(f"{self.cmd}\n")
        os.chmod(self.filenames.exec_filename, 0o777)

    def run(self):
        """Run job"""
        # Remove log file since it's used for tracking job progress
        with contextlib.suppress(OSError):
            os.remove(self.filenames.log_filename)
        # Run job
        schedd = htcondor.Schedd()
        with schedd.transaction() as txn:
            self.cluster_id = self.job.queue(txn)

    def cleanup(self, **kwargs):
        """Clean up intermediate files generated by job"""
        if not kwargs.get("cleanup", False):
            return
        with contextlib.suppress(OSError):
            for filename in self.filenames:
                os.remove(filename)

    @staticmethod
    def monitor(jobs, **kwargs):
        """Monitor jobs"""
        # pylint: disable = too-many-nested-blocks
        # TODO: Handle node failures by restarts before throwing exception
        # TODO: Don't wait for events indefinitely
        events = [htcondor.JobEventLog(job.filenames.log_filename).events(0) for job in jobs]
        num_unfinished_jobs = len(jobs)
        job_finished = [False] * num_unfinished_jobs
        while num_unfinished_jobs > 0:
            for idx in filter(lambda idx: not job_finished[idx], range(len(jobs))):
                job = jobs[idx]
                for event in events[idx]:
                    event_type = event.type
                    if event_type == JobEventType.JOB_TERMINATED:
                        if event["TerminatedNormally"]:
                            if event["ReturnValue"] != 0:
                                raise RuntimeError(f"Cmd: '{job.cmd}' terminated normally with non-zero return code - "
                                                   f"see error file: {job.filenames.err_filename}.")
                            job_finished[idx] = True
                            num_unfinished_jobs -= 1
                            job.cleanup()
                            break
                        # Terminated abnormally
                        CondorJobWrapper.remove_jobs(jobs, reason=f"Job {job.name} terminated abnormally")
                        raise RuntimeError(f"Cmd: '{job.cmd}' terminated abnormally - see log: {job.filenames.log_filename}.")
                    if event_type == JobEventType.JOB_ABORTED:
                        CondorJobWrapper.remove_jobs(jobs, reason=f"Job {job.name} aborted")
                        raise RuntimeError(f"Cmd: '{job.cmd}' aborted - see log: {job.filenames.log_filename}.")
            time.sleep(30)
        if jobs and jobs[0].shared_filesystem:
            time.sleep(30)  # Time to allow file changes to reflect in shared filesystem

    @staticmethod
    def remove_jobs(jobs, reason=None):
        """Remove jobs from condor queue"""
        schedd = htcondor.Schedd()
        for job in jobs:
            schedd.act(JobAction.Remove, f"ClusterId=={job.cluster_id}", reason=reason)
