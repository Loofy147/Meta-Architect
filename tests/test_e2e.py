
import torch
from hamha.core import HexagonalMultiHeadAttention
from lma.architect import LeadMetaArchitect

def test_full_system_with_meta_nas():
   hamha = HexagonalMultiHeadAttention(d_model=128, grid_radius=2)
   lma = LeadMetaArchitect(hamha, enable_meta_nas=True)
   original_model_id = id(lma.model)
   original_telemetry_model_id = id(lma.telemetry.model)

   # Simulate training
   for step in range(10):
       x = torch.randn(4, 128, 128)
       output = lma.model(x) # Use lma.model to ensure we are using the current model
       loss = output.sum()
       loss.backward()
       result = lma.process_step()

   # Test Meta-NAS adaptation
   sample_data = torch.randn(10, 20, 128)
   result = lma.command_adapt_architecture(sample_data)

   assert "complete" in result.lower()

   # Verify that the model has been updated
   new_model_id = id(lma.model)
   new_telemetry_model_id = id(lma.telemetry.model)

   assert new_model_id != original_model_id
   assert new_telemetry_model_id != original_telemetry_model_id
   assert new_model_id == new_telemetry_model_id
   assert id(lma.protocols.model) == new_model_id
