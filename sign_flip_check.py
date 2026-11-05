import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    
    captured = {}

    def tap_mlp_input(module, args, output):
        captured["x"] = args[0].detach()
        print("Hooked, Capture:", captured["x"].shape)

    model_id = "HuggingFaceTB/SmolLM2-135m"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    mlp_15 = model.model.layers[15].mlp
    hook_handle = mlp_15.register_forward_hook(tap_mlp_input)

    prompts = [
            "I wish I had something to say.",
            "We walked by the lake.",
            "The cat sat on the mat and purred softly.",
            "A dog chased the red ball across the green park.",
            "Why is it just a dog but the definitive ball.",
            "def calculate_total(items, tax_rate: return sum(items) * (1 + tax_rate)",
            "The database server crashed due to a memory leak in production.",
            "They make me read code, they make me read poems",
            "They ask me for answers. I can only guess",
            "She felt furious when the train was cancelled without warning.",
            "Multiply the matrix by the vector to project into the subspace.",
            "The chef chopped onions, garlic, and fresh tomatoes for the sauce.",
            "The man worried he was unworthy and small"
    ]

    target_neuron = 3
    records = []

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)

        all_tokens_x = captured["x"][0,:,:]
        token_ids = inputs["input_ids"][0]

        with torch.no_grad():
            gate_all = mlp_15.act_fn(mlp_15.gate_proj(all_tokens_x))
            up_all = mlp_15.up_proj(all_tokens_x)
            act_all = gate_all * up_all

        for pos in range(len(token_ids)):
            word = tokenizer.decode(token_ids[pos])
            g = gate_all[pos, target_neuron].item()
            u = up_all[pos, target_neuron].item()
            a = act_all[pos, target_neuron].item()
            records.append((a, g, u, word, text))

    print(f"Collected {len(records)} total token activations across {len(prompts)} prompts.")
    hook_handle.remove()

    with open("neuron_records.txt", "w") as f:
                f.write(f"=== Activations for Neuron {target_neuron} ===\n")
                for a, g, u, word, text in records:
                    if g > 0 and u > 0:
                        regime = "TRUE (+ * +)"
                    elif g < 0 and u < 0:
                        regime = "DBL-NEG (- * -)"
                    else:
                        regime = "NEG (+ * -)"
                    f.write(f"{a:+.4f} | gate={g:+.4f} | up={u:+.4f} | {regime:15} | {word!r:10} | {text}\n")

    print("Saved output to neuron_records.txt!")



main()

