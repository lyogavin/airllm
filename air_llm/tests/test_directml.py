import unittest
import torch

from airllm.device_utils import is_directml_available, get_directml_device


@unittest.skipUnless(is_directml_available(), "torch-directml not installed — skipping DirectML tests")
class TestDirectMLTensorOps(unittest.TestCase):
    """
    Validates that basic PyTorch tensor operations work correctly on an
    Intel / AMD integrated GPU via torch-directml.

    These tests are automatically skipped when torch-directml is not installed.
    Install with: pip install torch-directml
    """

    def setUp(self):
        self.device = get_directml_device(0)

    def test_device_string(self):
        self.assertIn("privateuseone", str(self.device))

    def test_tensor_creation_on_device(self):
        t = torch.randn(64, 64).to(self.device)
        self.assertEqual(str(t.device), "privateuseone:0")

    def test_matmul(self):
        a = torch.randn(512, 512).to(self.device)
        b = torch.randn(512, 512).to(self.device)
        c = torch.matmul(a, b)
        self.assertEqual(c.shape, torch.Size([512, 512]))
        self.assertEqual(str(c.device), "privateuseone:0")

    def test_float16_matmul(self):
        """AirLLM uses float16 by default — verify it works on iGPU."""
        a = torch.randn(256, 256, dtype=torch.float16).to(self.device)
        b = torch.randn(256, 256, dtype=torch.float16).to(self.device)
        c = torch.matmul(a, b)
        self.assertEqual(c.dtype, torch.float16)

    def test_layer_move_to_device(self):
        """Simulate AirLLM moving a transformer layer's weights to the iGPU."""
        layer_weights = {
            'self_attn.q_proj.weight': torch.randn(512, 512),
            'self_attn.k_proj.weight': torch.randn(512, 512),
            'mlp.gate_proj.weight':    torch.randn(1024, 512),
        }
        moved = {k: v.to(self.device) for k, v in layer_weights.items()}
        for name, tensor in moved.items():
            self.assertEqual(str(tensor.device), "privateuseone:0",
                             f"{name} not on iGPU")

    def test_layer_unload_to_cpu(self):
        """Simulate AirLLM unloading a layer back to CPU after forward pass."""
        t = torch.randn(512, 512).to(self.device)
        t_cpu = t.cpu()
        self.assertEqual(t_cpu.device.type, "cpu")
        self.assertEqual(t_cpu.shape, torch.Size([512, 512]))

    def test_airllm_layer_cycle(self):
        """
        Full cycle: CPU → iGPU (load) → compute → CPU (unload).
        Mirrors what AirLLM does for every transformer layer.
        """
        # Weights on CPU (as loaded from disk)
        weight = torch.randn(256, 256, dtype=torch.float16)
        x = torch.randn(1, 256, dtype=torch.float16)

        # Move to iGPU
        weight_gpu = weight.to(self.device)
        x_gpu = x.to(self.device)

        # Forward pass (linear layer equivalent)
        out = x_gpu @ weight_gpu.T

        self.assertEqual(str(out.device), "privateuseone:0")
        self.assertEqual(out.shape, torch.Size([1, 256]))

        # Unload weight (free iGPU memory)
        del weight_gpu
        result = out.cpu()
        self.assertEqual(result.device.type, "cpu")

    def test_multiple_sequential_layers(self):
        """
        Verify sequential layer processing works — each layer loaded, used,
        then freed, mimicking AirLLM's sharded inference loop.
        """
        x = torch.randn(1, 128, dtype=torch.float16)

        for i in range(5):
            layer_weight = torch.randn(128, 128, dtype=torch.float16)
            weight_gpu = layer_weight.to(self.device)
            x_gpu = x.to(self.device)
            x = (x_gpu @ weight_gpu.T).cpu()
            del weight_gpu, x_gpu

        self.assertEqual(x.shape, torch.Size([1, 128]))
        self.assertEqual(x.device.type, "cpu")
