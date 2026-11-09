import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Inspect SwiGLU sign splits for transformer neurons +*+ +*- -*-")
    parser.add_argument("--prompt", type=str, default=None, help="A single prompt to test, your choice!")
    parser.add_argument("--layer", type=int, default=15, help="Which Transformer Layer to hook, arr!")
    parser.add_argument("--neuron", type=int, default = 3, help="Which Neuron Index to Inspect, lookasee!")
    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
                "She sold sea shells by the sea shore",
                "Very late.",
                "One plus three take five... oh what do you want?",
                "Wherever you go, there you are.",
                "Consider this man for the job!",
                "It rises, it falls, it rhymes without repition.",
                "Have you ever actually listened to the ocean?",
                "Where are we?",
                "I think therefore I am.",
                "Real science will require better sample prompts.",
                "For now though, this is what we have",
                "Eight plus five is thirteen.",
                "8 + 5 = 13",
                "Eight + Five = Thirteen",
                "Here: Take this comma, may it serve you... as well as it served me!",
                "He updated the software to version one point one.",
                "Please update the software to version one.",
                "Machine, Update Software"
                ]

    target_layer = args.layer
    target_neuron = args.neuron

    print(f"Ok, here we go! Doing the thing. Model should load up. Lets look at layer  {target_layer} and Neuron {target_neuron}")

    captured = {}

    def tap_mlp_input(module, args, output):
        captured["x"] = args[0].detach()

    model_id = "HuggingFaceTB/SmolLM2-135m"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    mlp_block = model.model.layers[target_layer].mlp
    hook_handle = mlp_block.register_forward_hook (tap_mlp_input)
    
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

        for pos in range(len(token_ids)):
            word = tokenizer.decode(token_ids[pos])
            g = gate_all[pos, target_neuron].item()
            u = up_all[pos, target_neuron].item()
            a = act_all[pos, target_neuron].item()
            records.append((a, g, u, word, text))

    print(f"Collected {len(records)} total token activations across {len(prompts)} prompts.")
    hook_handle.remove()

    records.sort(key=lambda x: x[0], reverse=True)
    true_excitations = [r for r in records if r[1] > 0 and r[2] > 0]
    double_negatives = [r for r in records if r[1] < 0 and r[2] < 0]
    top_10_naive = records[:10]
    inversion_count = sum(1 for a, g, u, w, t in top_10_naive if g < 0 and u < 0)
    
    if args.prompt:
        prompt_label = prompts[0].split(maxsplit=1)[0].strip(",.!?:;")
    else:
        prompt_label = "DefaultCorpus"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("Records", exist_ok=True)
    filename = f"Records/{prompt_label}_L{target_layer}_N{target_neuron}_{timestamp}.txt"
    with open(filename, "w") as f:
                f.write(f"=== Results for Layer {target_layer} and  Neuron {target_neuron} ===\n")
                
                f.write(f"Top-10 Double Negative Inversions: {inversion_count}/10({inversion_count * 10}%)\n")
                
                f.write(f"--- Top Naive Activations (Unsorted Sign Mix) ---\n")
                for a, g, u, word, text in records[:10]:
                    f.write(f"{a:+.4f} | gate={g:+.4f} | up={u:+.4f} | {word!r:10} | {text}\n")
                
                f.write(f"\n--- Top True Excitations (+ * +) ---\n")
                for a, g, u, word, text in true_excitations[:10]:
                    f.write(f"{a:+.4f} | gate={g:+.4f} | up={u:+.4f} | {word!r:10} | {text}\n")
                
                f.write(f"\n--- Top Double-Negative Traps (- * -) ---\n")
                for a, g, u, word, text in double_negatives[:10]:
                    f.write(f"{a:+.4f} | gate={g:+.4f} | up={u:+.4f} | {word!r:10} | {text}\n")

    print(f"Saved structured output for {target_neuron} at {target_layer}!")

if __name__ == "__main__":
    main()

