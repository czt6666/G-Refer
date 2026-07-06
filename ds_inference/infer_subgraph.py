"""Inference for a Phase 3 soft-prompt-trained checkpoint.

ds_inference/infer.py doesn't know about the soft-prompt mechanism (it just
calls model.generate(input_ids=...)), so a checkpoint trained with
--subgraph_embed_path needs its generation calls built the same way training
built them: project the example's subgraph embedding and prepend it to
inputs_embeds. This script mirrors infer.py's I/O (reads {save_dir}/test.json,
writes {save_dir}/gen_datas.jsonl) but does that extra step.
"""
import argparse
import json
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,
                                             'subgraph_retriever')))
from subgraph_encoder import SubgraphProjector  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--subgraph_embed_path', required=True)
    p.add_argument('--subgraph_dim', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_tokens', type=int, default=256)
    p.add_argument('--save_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()

    state = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'),
                       map_location='cpu', weights_only=False)
    projector = SubgraphProjector(args.subgraph_dim, model.config.hidden_size)
    projector_state = {k[len('subgraph_projector.'):]: v for k, v in state.items()
                       if k.startswith('subgraph_projector.')}
    projector.load_state_dict(projector_state)
    projector = projector.to(model.device, dtype=model.dtype)
    projector.eval()

    embeds = torch.load(args.subgraph_embed_path, map_location='cpu', weights_only=False)
    zero_vec = torch.zeros(args.subgraph_dim)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    gen_datas_jsonl = save_dir / 'gen_datas.jsonl'

    rows = []
    with open(save_dir / 'test.json') as f:
        for line in f:
            rows.append(json.loads(line))

    with torch.inference_mode():
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i + args.batch_size]
            prompts = [r['prompt'] for r in batch]
            enc = tokenizer(prompts, padding=True, return_tensors='pt').to(model.device)

            sg = torch.stack([
                embeds.get(f"{r['uid']}-{r['iid']}", zero_vec) for r in batch
            ]).to(model.device, dtype=model.dtype)
            soft = projector(sg).unsqueeze(1)

            inputs_embeds = model.get_input_embeddings()(enc.input_ids)
            inputs_embeds = torch.cat([soft, inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [torch.ones(len(batch), 1, device=model.device, dtype=enc.attention_mask.dtype),
                 enc.attention_mask], dim=1)

            output_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=args.max_tokens, do_sample=False, temperature=0.0,
                    bos_token_id=tokenizer.bos_token_id,
                    pad_token_id=tokenizer.pad_token_id),
            )
            # With inputs_embeds, generate() returns only the newly generated
            # tokens (no prompt input_ids to prepend, since none were passed).
            output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            with open(gen_datas_jsonl, 'a') as f:
                for j, (row, out_str) in enumerate(zip(batch, output_strs)):
                    json.dump(dict(index=i + j, source_data=row,
                                  input_str=row['prompt'], output_str=out_str), f)
                    f.write('\n')


if __name__ == '__main__':
    main()
