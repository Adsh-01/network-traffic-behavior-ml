#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser(description="Network Traffic Behavior CLI")
    parser.add_argument("--mode", choices=["collect","train","eval"], default="collect")
    args = parser.parse_args()
    print(f"Mode: {args.mode}")
