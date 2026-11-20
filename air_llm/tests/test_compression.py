import sys
import unittest

import torch
sys.path.insert(0, '../airllm')

from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict




class TestCompression(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_should_compress_uncompress(self):
        #torch.manual_seed(0)
        a0 = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()
        a1 = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()

        a_state_dict = {'a0':a0, 'a1':a1}

        loss_fn = torch.nn.MSELoss()

        for iloop in range(10):
            for compression in [None, '4bit', '8bit']:
                b = compress_layer_state_dict(a_state_dict, compression)

                if iloop < 2:
                    print(f"for compression {compression}, compressed to: { {k:v.shape for k,v in b.items()} }")

                aa = uncompress_layer_state_dict(b)

                for k in aa.keys():

                    if compression is None:
                        self.assertTrue(torch.equal(aa[k], a_state_dict[k]))
                    else:
                        RMSE_loss = torch.sqrt(loss_fn(aa[k], a_state_dict[k])).detach().cpu().item()
                        print(f"compression {compression} loss: {RMSE_loss}")
                        self.assertLess(RMSE_loss, 0.1)

    def test_one_dim_tensors_stay_uncompressed(self):
        # norm weights / scalars are stored as is; matrices are quantized and come back close
        w = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()
        norm = (1 + 0.01 * torch.arange(128, dtype=torch.float16)).cuda()
        scalar = torch.tensor([0.5], dtype=torch.float16).cuda()
        state = {'mlp.w': w, 'mlp.w2': w.clone(), 'norm.weight': norm, 'layer_scalar': scalar}

        for compression in ['4bit', '8bit']:
            packed = compress_layer_state_dict(state, compression)
            self.assertTrue(torch.equal(packed['norm.weight'], norm))
            self.assertTrue(torch.equal(packed['layer_scalar'], scalar))
            self.assertNotEqual(packed['mlp.w'].dtype, w.dtype)

            out = uncompress_layer_state_dict(packed)
            self.assertEqual(set(out), set(state))
            self.assertTrue(torch.equal(out['norm.weight'], norm))
            self.assertTrue(torch.equal(out['layer_scalar'], scalar))
            for k in ('mlp.w', 'mlp.w2'):
                rmse = torch.sqrt(torch.nn.functional.mse_loss(out[k].float(), state[k].float())).item()
                self.assertLess(rmse, 0.1)

