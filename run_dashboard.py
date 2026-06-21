# run_dashboard.py
# Run this to start the Streamlit dashboard
# Command: python run_dashboard.py
#
# IMPORTANT: Keep run_step7.py running in another terminal!
# The dashboard calls the FastAPI backend.

import subprocess
import sys

print("=" * 55)
print("STEP 8: Starting Streamlit Dashboard")
print("=" * 55)
print("\n📊 Starting dashboard...")
print("   Dashboard URL: http://localhost:8501")
print("\n   ⚠️  Make sure API is running in another terminal!")
print("   If not: python run_step7.py\n")

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "dashboard/app.py",
    "--server.port", "8501",
    "--server.address", "localhost"
])