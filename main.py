import torch
import torch
from hamha.core import HexagonalMultiHeadAttention
from lma.architect import LeadMetaArchitect
from utils.visualization import plot_hexagonal_grid
import matplotlib.pyplot as plt


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
    lma.command_activate_module("GNN_OPT", {"target_t_mix": 35})
    lma.command_activate_module("ADAPT_BIAS", {"mode": "exploration"})

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
            status = result["status"]
            print(f"Health: {status['health']}")
            print(f"Avg Entropy: {status['avg_entropy']:.3f}")
            print(f"Max κ: {status['max_kappa']:.2f}")
            print(f"Alerts: {status['active_alerts']}")
            if result["interventions"]:
                print(f"Interventions: {', '.join(result['interventions'])}")

    # Generate final report
    print("\n" + lma.generate_report())

    return hamha, lma


if __name__ == "__main__":
    # In main_demo, LMA is instantiated but Meta-NAS is not enabled by default.
    # Let's create a new run that enables it.
    print("\n--- Running Standard LMA Demo ---")
    main_demo()

    print("\n\n--- Running LMA Demo with Meta-NAS Enabled ---")
    d_model = 512
    grid_radius = 2
    hamha_nas = HexagonalMultiHeadAttention(d_model=d_model, grid_radius=grid_radius)
    lma_nas = LeadMetaArchitect(hamha_nas, enable_meta_nas=True)

    # Simulate a few steps
    for step in range(5):
        x = torch.randn(4, 128, d_model)
        output = lma_nas.model(x)
        loss = output.sum()
        loss.backward()
        lma_nas.process_step()

    print(lma_nas.generate_report())

    # Now, trigger adaptation
    print("\n--- Triggering Meta-NAS Adaptation ---")
    new_task_data = torch.randn(8, 64, d_model)
    original_arch = {
        "grid_radius": max(abs(c.q) for c in lma_nas.model.grid_coords),
        "num_heads": lma_nas.model.num_heads
    }
    print(f"Original architecture: {original_arch}")
    fig = plot_hexagonal_grid(lma_nas.model, "Original Architecture")
    fig.savefig("original_architecture.png")
    print("Saved original architecture plot to original_architecture.png")

    result = lma_nas.command_adapt_architecture(new_task_data)
    print(f"LMA Response: {result}")

    new_arch = {
        "grid_radius": max(abs(c.q) for c in lma_nas.model.grid_coords),
        "num_heads": lma_nas.model.num_heads
    }
    print(f"New architecture: {new_arch}")
    fig = plot_hexagonal_grid(lma_nas.model, "Adapted Architecture")
    fig.savefig("adapted_architecture.png")
    print("Saved adapted architecture plot to adapted_architecture.png")

    print("\n✓ Meta-NAS demonstration complete.")
