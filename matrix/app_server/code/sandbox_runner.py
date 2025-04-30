# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import base64
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SandboxRunner:
    """
    A wrapper around bubblewrap to safely execute Python code in a sandbox.

    This class creates a secure environment using bubblewrap where Python code can be
    executed with controlled access to resources and filesystem.
    """

    def __init__(
        self,
        shared_paths: List[Path],
        resource_limits: Dict[str, int],
    ):
        """
        Initialize the sandbox runner with shared paths and resource limits.

        Args:
            shared_paths: List of Path objects that should be mounted into the sandbox
            resource_limits: Dictionary of resource limits (memory, cpu, etc.)
            conda_prefix: Path to conda env
        """
        self.shared_paths = shared_paths
        self.resource_limits = resource_limits

        # Check if bubblewrap is installed
        try:
            subprocess.run(
                ["bwrap", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "bubblewrap (bwrap) is not installed or not available in PATH"
            )

        # Use the system Python if no conda environment specified
        self.python_path = Path(
            subprocess.check_output(["which", "python3"]).decode().strip()
        )

    def run(self, code: str) -> Dict[str, Any]:
        """
        Run the provided Python code in a sandbox.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with keys:
                - error_msg: Error message if any
                - success: Boolean indicating success
                - result: Captured output or None if there was an error
        """
        result = {"error_msg": "", "success": False, "result": None}

        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write code to a file
            code_file = Path(temp_dir) / "code.py"
            result_file = Path(temp_dir) / "result.json"

            # Simplified wrapper to capture stdout and handle errors
            encoded_code = base64.b64encode(code.encode()).decode()
            wrapped_code = f"""
import sys
import traceback
import json
import base64
from io import StringIO

# Capture stdout
output_capture = StringIO()
original_stdout = sys.stdout
sys.stdout = output_capture

result = {{
    "error_msg": "",
    "success": False,
    "result": None
}}

try:
    # Execute the code
    code_to_execute = "{encoded_code}"
    decoded_code = base64.b64decode(code_to_execute).decode('utf-8')
    exec(decoded_code)
    
    # Execution was successful
    result["success"] = True
    
    # Get captured output
    sys.stdout = original_stdout
    result["result"] = output_capture.getvalue()
    
except Exception as e:
    # Restore stdout
    sys.stdout = original_stdout
    
    # Record the error
    result["error_msg"] = f"{{type(e).__name__}}: {{str(e)}}\\n{{traceback.format_exc()}}"

# Write result to file
with open('{result_file}', 'w') as f:
    json.dump(result, f)
"""

            with open(code_file, "w") as f:
                f.write(wrapped_code)

            # Build bubblewrap command
            bwrap_cmd = ["bwrap", "--die-with-parent"]

            # Add filesystem isolation
            bwrap_cmd.extend(["--unshare-all", "--share-net"])

            # Bind necessary system paths
            system_paths = [
                "/usr",
                "/lib",
                "/lib64",
                "/bin",
                "/sbin",
                "/etc/alternatives",
                "/etc/ssl",
            ]

            for path in [Path(p) for p in system_paths]:
                if path.exists():
                    bwrap_cmd.extend(["--ro-bind", str(path), str(path)])

            # If using conda, bind the conda installation and environment
            # Just bind the system Python
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                conda_path = conda_prefix.split("/envs/")[0]
                bwrap_cmd.extend(["--ro-bind", str(conda_path), str(conda_path)])
            bwrap_cmd.extend(
                ["--ro-bind", str(self.python_path), str(self.python_path)]
            )

            # Bind user-specified paths
            for path in self.shared_paths:
                if path.exists():
                    target_path = path
                    bwrap_cmd.extend(["--ro-bind", str(path), str(target_path)])

            # Bind the temp directory
            bwrap_cmd.extend(["--bind", temp_dir, temp_dir])

            # CPU time limit in seconds
            timeout = self.resource_limits.get("time", 30)  # Default: 30 seconds

            # Finally, add the command to execute
            bwrap_cmd.extend([str(self.python_path), str(code_file)])

            try:
                # Run the sandboxed code with timeout
                process = subprocess.Popen(bwrap_cmd)

                # Wait for process to complete or timeout
                start_time = time.time()
                while process.poll() is None:
                    if time.time() - start_time > timeout:
                        # Kill the process if it exceeds the timeout
                        process.kill()
                        result["error_msg"] = (
                            f"Execution timed out after {timeout} seconds"
                        )
                        return result
                    time.sleep(0.1)

                # Check if result file exists and read it
                if result_file.exists():
                    with open(result_file, "r") as f:
                        try:
                            return json.load(f)
                        except json.JSONDecodeError:
                            result["error_msg"] = "Failed to decode result JSON"
                else:
                    result["error_msg"] = "Execution failed to produce a result file"

            except subprocess.SubprocessError as e:
                result["error_msg"] = f"Subprocess error: {str(e)}"
            except Exception as e:
                result["error_msg"] = f"Unexpected error: {str(e)}"

        return result
