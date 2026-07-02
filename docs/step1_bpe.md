# Step 1: BPE Tokenizer

Step 1 replaces the character-level tokenizer with byte-level BPE while keeping
the model shape, seed, dataset source, and training budget fixed.

## Why This Matters

Character-level tokenization is useful for Step 0 because it is transparent and
cannot produce unknown tokens. Real GPT-style models use subword tokenization,
which gives a much better compute/text tradeoff:

- common words or fragments can become single tokens
- rare text can still be represented through smaller byte-level pieces
- sequences become shorter than character-level sequences
- the project moves closer to how GPT-2 and modern LLMs are actually trained

## Fair Comparison Rules

Do not compare raw token-level loss alone and call BPE "better." The token unit
has changed. Report at least:

- validation loss
- generated samples from the same prompt
- tokens per second
- train/validation token counts
- average characters per token

## Run

On the Windows training machine:

```bash
python data/prepare_bpe.py --vocab_size 8000
python train.py --config configs/bpe_5070.yaml
python eval/generate.py --out_dir out/bpe_5070 --start "ROMEO:" --output_file out/bpe_5070/samples.txt
python scripts/plot_log.py --log out/bpe_5070/log.csv --out out/bpe_5070/loss_curve.png
python scripts/export_run.py --out_dir out/bpe_5070
```

Export `reports/runs/bpe_5070/`, not the checkpoint.
