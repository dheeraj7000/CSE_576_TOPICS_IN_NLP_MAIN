#!/usr/bin/env python3
"""
Simple runner for LogiQA baseline evaluation
"""

import subprocess
import sys
import os

def find_python():
    """Find available Python executable"""
    candidates = [
        '/opt/miniconda3/bin/python',
        '/usr/bin/python3',
        '/usr/bin/python',
        'python3',
        'python'
    ]
    
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return candidate
        except:
            continue
    
    return None

def main():
    python_exe = find_python()
    if not python_exe:
        print("No Python executable found")
        return 1
    
    print(f"Using Python: {python_exe}")
    
    # Run the 4GB optimized version with sample data
    cmd = [python_exe, 'logiqa_4gb.py', '--max_samples', '5']
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        return result.returncode
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())