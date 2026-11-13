import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model (model_id: str):
    print(f"Getting Model: {model_id} for you now! We'll use a local copy if we find one")
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    print(f"Onward! Load Model Complete and returning Model and Tokenizer to Main Function Now")
    return model, tokenizer

def get_vector(word, tokenizer, embed_matrix):
    token_id = tokenizer.encode(word, add_special_tokens=False)[0]
    return embed_matrix[token_id]


def analyze_neuron(model, tokenizer, prompts, target_layer, target_neuron):
    
    captured = {}
    def tap_mlp_input(module, args, output):
        captured["x"] = args[0].detach()


    mlp_block = model.model.layers[target_layer].mlp    
    hook_handle = mlp_block.register_forward_hook(tap_mlp_input)

    records = []
    prompt_batches = []

    print(f"working on layer {target_layer}, prompts are going in...")
    
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

        words = [tokenizer.decode(t) for t in token_ids]
        prompt_batches.append((act_all, gate_all, up_all, words, text))

    hook_handle.remove()
    print(f"Hook Handle Removed. Layer {target_layer} complete. Looping over Neurons now")

    layer_results = []

    for n in target_neuron:
        records = []
        print(f"Peeking at Neuron {n}") 
        for act_all, gate_all, up_all, words, text in prompt_batches:
            for pos in range(len(words)):
                g = gate_all[pos, n].item()
                u = up_all[pos, n].item()
                a = act_all[pos, n].item()
                records.append((a, g, u, words[pos], text))

        records.sort(key=lambda x: x[0], reverse=True)
        true_exc = [r for r in records if r[1] > 0 and r[2] > 0]
        dbl_neg = [r for r in records if r[1] < 0 and r[2] < 0]
        top_10 = records[:10]
        inv_count = sum(1 for a, g, u, w, t, in top_10 if g < 0 and u < 0)

        max_true_act = round(true_exc[0][0], 4) if true_exc else 0.0
        top_true_word = true_exc[0][3] if true_exc else "None"
        max_dbl_act = round(dbl_neg[0][0], 4) if dbl_neg else 0.0
        top_dbl_word = dbl_neg[0][3] if dbl_neg else "None"
        
        layer_results.append([
            target_layer, n, inv_count, inv_count * 10,
            max_true_act, top_true_word, max_dbl_act, top_dbl_word
            ])

    return layer_results
