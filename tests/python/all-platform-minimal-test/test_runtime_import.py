# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TVM import behavior tests."""

import errno
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX job control")


def _run_in_pty(script):
    import pty

    pid, output_fd = pty.fork()
    if pid == 0:
        os.execl(sys.executable, sys.executable, "-c", script)

    output = bytearray()
    while True:
        try:
            chunk = os.read(output_fd, 4096)
        except OSError as err:
            if err.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        output.extend(chunk)

    _, status = os.waitpid(pid, 0)
    os.close(output_fd)
    assert os.waitstatus_to_exitcode(status) == 0, output.decode(errors="replace")


def test_import_preloads_readline_for_foreground_tty():
    _run_in_pty(
        """
import importlib.util
import sys

assert "readline" not in sys.modules
readline_is_available = importlib.util.find_spec("readline") is not None
import tvm
assert ("readline" in sys.modules) == readline_is_available
import pdb
pdb.Pdb()
"""
    )


def test_import_does_not_stop_background_process_group():
    _run_in_pty(
        """
import os
import signal
import sys

for redirect_stdin in (False, True):
    pid = os.fork()
    if pid == 0:
        os.setpgid(0, 0)
        if redirect_stdin:
            stdin_fd = os.open(os.devnull, os.O_RDONLY)
            os.dup2(stdin_fd, 0)
            os.close(stdin_fd)
        import tvm
        assert "readline" not in sys.modules
        os._exit(0)

    _, status = os.waitpid(pid, os.WUNTRACED)
    if os.WIFSTOPPED(status):
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise AssertionError(f"TVM import stopped by signal {os.WSTOPSIG(status)}")
    if os.waitstatus_to_exitcode(status) != 0:
        raise AssertionError(f"TVM import exited with status {status}")
"""
    )
