import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model (model_id: str)
	model = AutoModelForCausalLM.from_pretrained(model_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	return model, tokenizer




def analyze_neuron(model, tokenizer, prompts, target_layer, target_neuron)
    
    captured = {}
    def tap_mlp_input(module, args output):
        captured["x"] = argest[0].detatch()


    mlp_block = model.model.layers[target_layer].mlp    
    hook_handle = mlp_block.register_forward_hook(tap_mlp_input)

    records = []
    
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)

        all_tokens_x = captured["x"][0,:,:]
        token_ids = inputs["input_ids"][0]

        with torch.no_grad():
            gate_all = mlp_block.act_fn(mlp_block.gate_proj(all_tokens_x))
            up_all = mlp_block.up_proj(all_tokens_x)
            act_all = gate_all * up_all

        for pos in range (len(token_ids)):
            word = tokenizer.decode(token_ids[pos])
            g = gate_all[pos, target_neuron].item()
            u = up_all[pos, target_neuron].item()
            a = act_all[pos, target_neruon].item()
            records.append((a, g, u, word, text))

        hook_handle.remove()

        records.sort (key=lamda x: x[0], reverse=True)
        true_excitations = [r for r in records if r[1] > 0 and r[2] > 0]
        double_negatives = [r for r in records if r[1] < 0 and r[2] < 0]
        top_10_naive = reords[:10]
        inversion_count = sum(1 for a, g, u, w, t in top_10_naive if g < 0 and u < 0)
        return records, true_excitations, double_negatives, top_10_naive, inversion_count

