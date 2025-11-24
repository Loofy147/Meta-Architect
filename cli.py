import argparse
import json
from main import main_demo
from examples.training_integration import train_with_lma_monitoring

def main_cli():
    """Command-line interface for HAMHA + LMA."""
    parser = argparse.ArgumentParser(
        description='HAMHA with Lead Meta-Architect Governance'
    )

    parser.add_argument('--mode', choices=['demo', 'train', 'analyze'],
                       default='demo', help='Execution mode')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--grid-radius', type=int, default=2,
                       help='Hexagonal grid radius')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of training steps')
    parser.add_argument('--checkpoint', type=str,
                       help='Load from checkpoint')
    parser.add_argument('--save-telemetry', action='store_true',
                       help='Save telemetry to file')

    args = parser.parse_args()

    if args.mode == 'demo':
        main_demo()

    elif args.mode == 'train':
        train_with_lma_monitoring()

    elif args.mode == 'analyze':
        print("Analysis mode will be implemented in a future version.")
        print("This mode will allow loading telemetry data and generating reports.")

    print("\n✓ Execution complete")


if __name__ == "__main__":
    main_cli()
