#!/usr/bin/env python3
import argparse
import sys, os

sys.path.append(os.path.dirname(__file__))

from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Network Traffic Behavior CLI")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", action="store_true", help="Evaluate model")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.eval:
        evaluate_model()
    else:
        print("✅ Commands:")
        print("python3 src/cli.py --train")
        print("python3 src/cli.py --eval")

if __name__ == "__main__":
    main()
