from src.demo import run_demo
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BearLLM on official or custom vibration data.')
    parser.add_argument('--data', help='Input JSON; defaults to MBHM_DATASET/demo_data.json in .env.')
    parser.add_argument('--checkpoint', help='BearLLM adapter directory; defaults to official weights in .env.')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', help='Save the generated response as a UTF-8 text file.')
    args = parser.parse_args()
    run_demo(data_file=args.data, max_new_tokens=args.max_new_tokens,
             seed=args.seed, output_file=args.output, checkpoint_dir=args.checkpoint)
