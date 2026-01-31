"""
Quick fix for Prophet installation - installs CmdStan backend.
Run this if you get "Prophet has no attribute 'stan_backend'" error.
"""

import subprocess
import sys

print("🔧 Installing CmdStan backend for Prophet...")
print("This may take 2-3 minutes...\n")

try:
    # Install cmdstanpy if not already installed
    print("Step 1: Installing cmdstanpy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cmdstanpy==1.2.0"])
    print("✅ cmdstanpy installed\n")
    
    # Install CmdStan
    print("Step 2: Installing CmdStan compiler...")
    subprocess.check_call([sys.executable, "-m", "cmdstanpy.install_cmdstan"])
    print("✅ CmdStan installed\n")
    
    print("=" * 60)
    print("✅ Prophet backend installation complete!")
    print("=" * 60)
    print("\nNow restart the Streamlit dashboard:")
    print("1. Stop the current dashboard (Ctrl+C)")
    print("2. Run: streamlit run app.py")
    print("\nProphet should now work correctly!")
    
except Exception as e:
    print(f"\n❌ Error during installation: {str(e)}")
    print("\nManual fix:")
    print("1. Run: pip install cmdstanpy")
    print("2. Run: python -m cmdstanpy.install_cmdstan")
    print("3. Restart dashboard")
