import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="WAV to MIDI converter")
    parser.add_argument("wav_path", type=str, help="Path to the input WAV file")
    args = parser.parse_args()

    return args.wav_path
