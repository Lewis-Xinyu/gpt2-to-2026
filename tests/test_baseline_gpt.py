import unittest

import torch

from model.baseline_gpt import GPT, GPTConfig, RMSNorm, SwiGLUMLP
from train import build_gpt_config


def tiny_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )


def tiny_rope_config() -> GPTConfig:
    cfg = tiny_config()
    cfg.position_embedding = "rope"
    return cfg


def tiny_rmsnorm_config() -> GPTConfig:
    cfg = tiny_rope_config()
    cfg.norm_type = "rmsnorm"
    return cfg


def tiny_swiglu_config() -> GPTConfig:
    cfg = tiny_rmsnorm_config()
    cfg.mlp_type = "swiglu"
    return cfg


class BaselineGPTTest(unittest.TestCase):
    def test_forward_returns_full_logits_and_loss(self):
        torch.manual_seed(1337)
        model = GPT(tiny_config())
        idx = torch.randint(0, model.config.vocab_size, (4, 8))
        targets = torch.randint(0, model.config.vocab_size, (4, 8))

        logits, loss = model(idx, targets)

        self.assertEqual(logits.shape, (4, 8, model.config.vocab_size))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_inference_returns_last_position_logits(self):
        model = GPT(tiny_config())
        idx = torch.randint(0, model.config.vocab_size, (2, 5))

        logits, loss = model(idx)

        self.assertEqual(logits.shape, (2, 1, model.config.vocab_size))
        self.assertIsNone(loss)

    def test_attention_is_causal(self):
        torch.manual_seed(1337)
        model = GPT(tiny_config()).eval()
        left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]) % model.config.vocab_size
        right = left.clone()
        right[:, 4:] = torch.tensor([[9, 10, 11, 12]]) % model.config.vocab_size

        left_logits, _ = model(left, left)
        right_logits, _ = model(right, right)

        self.assertTrue(torch.allclose(left_logits[:, :4], right_logits[:, :4], atol=1e-5))

    def test_weight_tying(self):
        model = GPT(tiny_config())
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_generate_extends_sequence(self):
        torch.manual_seed(1337)
        model = GPT(tiny_config()).eval()
        idx = torch.tensor([[1, 2, 3]])

        out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=4)

        self.assertEqual(out.shape, (1, 8))

    def test_rope_removes_learned_position_table(self):
        model = GPT(tiny_rope_config())
        self.assertNotIn("wpe", model.transformer)

    def test_rope_forward_and_causality(self):
        torch.manual_seed(1337)
        model = GPT(tiny_rope_config()).eval()
        idx = torch.randint(0, model.config.vocab_size, (2, 8))

        logits, loss = model(idx, idx)

        self.assertEqual(logits.shape, (2, 8, model.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

        left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]) % model.config.vocab_size
        right = left.clone()
        right[:, 4:] = torch.tensor([[9, 10, 11, 12]]) % model.config.vocab_size
        left_logits, _ = model(left, left)
        right_logits, _ = model(right, right)
        self.assertTrue(torch.allclose(left_logits[:, :4], right_logits[:, :4], atol=1e-5))

    def test_rmsnorm_forward(self):
        model = GPT(tiny_rmsnorm_config()).eval()
        idx = torch.randint(0, model.config.vocab_size, (2, 8))

        logits, loss = model(idx, idx)

        self.assertEqual(logits.shape, (2, 8, model.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsInstance(model.transformer.ln_f, RMSNorm)

    def test_train_config_builder_uses_new_defaults(self):
        cfg = build_gpt_config(
            {
                "block_size": 8,
                "vocab_size": 16,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 16,
                "dropout": 0.0,
                "bias": True,
            }
        )

        self.assertEqual(cfg.position_embedding, "learned")
        self.assertEqual(cfg.norm_type, "layernorm")
        self.assertEqual(cfg.mlp_type, "gelu")

    def test_swiglu_forward(self):
        model = GPT(tiny_swiglu_config()).eval()
        idx = torch.randint(0, model.config.vocab_size, (2, 8))

        logits, loss = model(idx, idx)

        self.assertEqual(logits.shape, (2, 8, model.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsInstance(model.transformer.h[0].mlp, SwiGLUMLP)


if __name__ == "__main__":
    unittest.main()
