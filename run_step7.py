# run_step7.py
# Starts the FastAPI server
# Command: python run_step7.py

import uvicorn

if __name__ == "__main__":
    print("=" * 55)
    print("STEP 7: Starting FastAPI Backend")
    print("=" * 55)
    print("\n📡 Starting server...")
    print("   API URL:  http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs  ← Open this!")
    print("   Health:   http://localhost:8000/health")
    print("\n   Press Ctrl+C to stop the server\n")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # ✅ FIX: disabled to avoid Windows multiprocessing issue
        log_level="info"
    )