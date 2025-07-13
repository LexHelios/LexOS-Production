#!/usr/bin/env python3
import os
import sys
import subprocess

# Set environment variable to disable OpenTelemetry if problematic
os.environ['OTEL_SDK_DISABLED'] = 'true'

# Run the main application
subprocess.run([sys.executable, "main.py"])