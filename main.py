import torch
from hamha.core import HexagonalMultiHeadAttention
from lma.architect import LeadMetaArchitect

def main_demo():
    """Demonstration of HAMHA with LMA governance."""

    print("\n" + "═" * 70)
    print("INITIALIZING HAMHA + LMA SYSTEM")
    print("═" * 70 + "\n")

    # Initialize HAMHA
    d_model = 512
    grid_radius = 2
    hamha = HexagonalMultiHeadAttention(d_model=d_model, grid_radius=grid_radius)

    # Initialize LMA
    lma = LeadMetaArchitect(hamha)

    # Activate evolutionary modules
    lma.command_activate_module('GNN_OPT', {'target_t_mix': 35})
    lma.command_activate_module('ADAPT_BIAS', {'mode': 'exploration'})

    print("\n" + "═" * 70)
    print("SIMULATING TRAINING STEPS")
    print("═" * 70 + "\n")

    # Simulate training steps
    for step in range(30):
        # Create dummy input
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        output = hamha(x)

        # Simulate backward (create fake gradients)
        loss = output.sum()
        loss.backward()

        # LMA processing
        result = lma.process_step()

        # Print status every 10 steps
        if step % 10 == 0:
            print(f"\n{'─' * 70}")
            print(f"STEP {step}")
            print(f"{'─' * 70}")
            status = result['status']
            print(f"Health: {status['health']}")
            print(f"Avg Entropy: {status['avg_entropy']:.3f}")
            print(f"Max κ: {status['max_kappa']:.2f}")
            print(f"Alerts: {status['active_alerts']}")
            if result['interventions']:
                print(f"Interventions: {', '.join(result['interventions'])}")

    # Generate final report
    print("\n" + lma.generate_report())

    return hamha, lma


if __name__ == "__main__":
    hamha_model, lma = main_demo()
    print("\n✓ HAMHA + LMA system operational and ready for deployment")
