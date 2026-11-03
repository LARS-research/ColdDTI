"""Create a reproducible frozen protein encoder with the nine structural tokens."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=57)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(args.seed)
    checkpoint = args.model_root / 'prot_bert'
    destination = args.output_dir or args.model_root / 'prot_resize'
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModel.from_pretrained(str(checkpoint), local_files_only=True)
    tokens = ['[H]', '[E]', '[T]', '[B]', '[C]', '[secondary_start]', '[secondary_end]',
              '[tertiary_start]', '[tertiary_end]']
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.eval().requires_grad_(False)
    tokenizer.save_pretrained(str(destination))
    model.save_pretrained(str(destination))
    (destination / 'structure_tokens.json').write_text(json.dumps(dict(seed=args.seed,
        token_ids=dict(zip(tokens, tokenizer.convert_tokens_to_ids(tokens)))), indent=2) + '\n')
    print('Saved expanded frozen encoder to', destination)


if __name__ == '__main__':
    main()
