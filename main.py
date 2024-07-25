import argparse
from tortoise_tts.train import train_model
from tortoise_tts.inference import text_to_speech

def main():
    parser = argparse.ArgumentParser(description="Tortoise TTS Script")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help="Mode: 'train' or 'inference'")
    parser.add_argument('--text', type=str, help="Text for inference")

    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'inference':
        if args.text:
            speech = text_to_speech(args.text)
            # Save or play the speech output
        else:
            print("Please provide text for inference.")
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
