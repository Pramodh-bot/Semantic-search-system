#!/usr/bin/env python3
"""
Quick Start Script - Execute this to get the system running

Run this script to quickly see if everything is configured correctly
and get the next steps.
"""

import sys
import os
import subprocess
import platform

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_step(number, title):
    """Print a numbered step."""
    print(f"\n{number}️⃣  {title}")
    print("-" * 70)


def check_python_version():
    """Check if Python 3.11+ is installed."""
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_venv():
    """Check if virtual environment is active."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  Virtual environment not active")
        print("\nTo activate:")
        if platform.system() == "Windows":
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        return False
    
    print("✅ Virtual environment is active")
    return True


def main():
    """Main execution."""
    print_header("SEMANTIC SEARCH SYSTEM - QUICK START")
    
    print("This script will guide you through setting up and running the system.")
    print()
    
    # Step 1: Check Python
    print_step(1, "Check Python Installation")
    if not check_python_version():
        print("\n❌ Please install Python 3.11 or later from python.org")
        return 1
    
    # Step 2: Check virtual environment
    print_step(2, "Check Virtual Environment")
    in_venv = check_venv()
    
    if not in_venv:
        response = input("\nCreate and activate virtual environment now? (y/n): ").strip().lower()
        if response == 'y':
            print("\nCreating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=PROJECT_ROOT)
            
            if platform.system() == "Windows":
                print("\n✅ Created! Now run: venv\\Scripts\\activate")
            else:
                print("\n✅ Created! Now run: source venv/bin/activate")
            
            print("\nThen run this script again.")
            return 0
        else:
            print("\n❌ Virtual environment required. Please activate it and try again.")
            return 1
    
    # Step 3: Check dependencies
    print_step(3, "Check Dependencies")
    
    try:
        import fastapi
        import uvicorn
        import numpy
        import sentence_transformers
        import faiss
        import sklearn
        print("✅ All required dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstalling dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      cwd=PROJECT_ROOT)
        print("\n✅ Dependencies installed. Running verification...")
        return main()  # Rerun this script
    
    # Step 4: Check if dataset is prepared
    print_step(4, "Check Dataset")
    
    vector_db_path = os.path.join(PROJECT_ROOT, "data", "vector_db.faiss")
    clustering_path = os.path.join(PROJECT_ROOT, "data", "clustering_model.pkl")
    
    if os.path.exists(vector_db_path) and os.path.exists(clustering_path):
        print("✅ Dataset and models already prepared")
    else:
        print("⚠️  Dataset not yet prepared")
        response = input("\nPrepare dataset now? This takes 5-10 minutes. (y/n): ").strip().lower()
        
        if response == 'y':
            print("\nDownloading 20 Newsgroups and preparing indices...")
            print("This may take 5-10 minutes on first run...")
            print()
            
            result = subprocess.run(
                [sys.executable, "src/download_dataset.py"],
                cwd=PROJECT_ROOT
            )
            
            if result.returncode != 0:
                print("\n❌ Dataset preparation failed")
                return 1
            
            print("\n✅ Dataset prepared successfully!")
        else:
            print("\nYou can prepare data later with:")
            print("  python src/download_dataset.py")
    
    # Step 5: Verification
    print_step(5, "Run Verification")
    print("Running component verification...")
    print()
    
    result = subprocess.run(
        [sys.executable, "verify_setup.py"],
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print("\n❌ Verification failed")
        return 1
    
    # Success!
    print_header("✅ SETUP COMPLETE!")
    
    print("""
Ready to start the API server!

Next command:
  python -m uvicorn src.api:app --reload

Then open in your browser:
  http://localhost:8000/docs

You can also run the demo:
  python test_demo.py

For detailed setup instructions, read:
  GETTING_STARTED.md

For architecture explanation, read:
  ARCHITECTURE.md
""")
    
    response = input("\nStart API server now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nStarting Uvicorn server...")
        print("Server will be available at http://localhost:8000")
        print("API documentation at http://localhost:8000/docs")
        print("\nPress CTRL+C to stop the server.\n")
        
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "src.api:app", "--reload"],
            cwd=PROJECT_ROOT
        )
    else:
        print("\nWhen ready, start the server with:")
        print("  python -m uvicorn src.api:app --reload")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
