import subprocess
import sys

if __name__ == "__main__":
    # Launch the Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "rag_V-7.py"])
