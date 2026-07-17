import sys
from pathlib import Path

# Add the project root to sys.path to ensure src can be imported
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.indexer import index_audio
except ImportError as e:
    print(f"Error: Could not import index_audio from src/indexer.py: {e}")
    sys.exit(1)

def main():
    print("=== Audio Separation Dataset Path Setup ===")
    print("This script will prompt for absolute Windows directory paths for Speech and Noise data,")
    print("and then generate indices pointing directly to these paths to bypass Windows symlink restrictions.\n")

    # Prompt user for Speech data path
    speech_dir_str = input("Please paste the absolute path for the SPEECH data: ").strip()
    # Strip quotes if they were pasted with quotes
    speech_dir_str = speech_dir_str.strip('"\'')
    speech_path = Path(speech_dir_str)
    if not speech_path.exists():
        print(f"Error: The Speech data directory does not exist: {speech_dir_str}")
        sys.exit(1)
    if not speech_path.is_dir():
        print(f"Error: The Speech data path is not a directory: {speech_dir_str}")
        sys.exit(1)

    # Prompt user for Noise data path
    noise_dir_str = input("Please paste the absolute path for the NOISE data: ").strip()
    # Strip quotes if they were pasted with quotes
    noise_dir_str = noise_dir_str.strip('"\'')
    noise_path = Path(noise_dir_str)
    if not noise_path.exists():
        print(f"Error: The Noise data directory does not exist: {noise_dir_str}")
        sys.exit(1)
    if not noise_path.is_dir():
        print(f"Error: The Noise data path is not a directory: {noise_dir_str}")
        sys.exit(1)

    print("\nPaths validated successfully. Starting index generation...")

    # Define destination output JSON indices
    speech_output = "data/indices/speech_index.json"
    noise_output = "data/indices/noise_index.json"

    # Run the indexer for Speech data
    print(f"\nIndexing Speech data from: {speech_path.resolve()}")
    try:
        index_audio(str(speech_path.resolve()), speech_output)
    except Exception as e:
        print(f"Error indexing Speech data: {e}")
        sys.exit(1)

    # Run the indexer for Noise data
    print(f"\nIndexing Noise data from: {noise_path.resolve()}")
    try:
        index_audio(str(noise_path.resolve()), noise_output)
    except Exception as e:
        print(f"Error indexing Noise data: {e}")
        sys.exit(1)

    print("\n=== Setup completed successfully! ===")

if __name__ == "__main__":
    main()
