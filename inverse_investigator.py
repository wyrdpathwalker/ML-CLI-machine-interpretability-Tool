import os
import argparse
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from model_analyze import load_model, analyze_neuron

def main():
    parser = argparse.ArgumentParser(description="Inspect SwiGLU sign splits")
    parser.add_argument ("--prompt", type=str, default=None, help="Enter a single custom prompt to test")
    parser.add_argument("--neuron", type=int, default=None, help="Pick a specific neuron. Default will run a loop over all neurons!")
    parser.add_argument("--layer", type=int, default=15, help="Pick the transormer layer you'd like to test, default is 15")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Enter a specific model to use. Default is SmolLM2-135m")
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
                   "Here: Take this comma, may it serve you well!",
                   "He updated the software to version one point one",
                   "Please update the software to version one.",
                   "Machine, Update Software"
                   ]

    if args.neuron is None:
        target_neuron = range(20)
    else:
        target_neuron = [args.neuron]

        #The Full Range for the default model is 1536 
        
        print(f"Howdy, welcome. Calling model and running checks. Gathering results for Layer {target_layer}.")

#Call The model here, once.
    model, tokenizer = load_model(args.model)

#Create a CSV file, First we build the file name parts, then make a directory called Records, then combine the filename in the directory. Then we call open, and use pythons csv writer.
    if args.prompt:
        prompt_label = prompts[0].split(maxsplit=1)[0].strip("!?,.:;")
    else: 
        prompt_label = "DefaultCorpus"
    
    target_layer = args.layer
    layer_label = args.layer

    neuron_label = args.neuron if args.neuron is not None else "all"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("Records", exist_ok=True)
    filename = f"Records/{prompt_label}_L{layer_label}_N{neuron_label}_TestDateTime{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Neuron", "Top10_Inversions", "Inversion_Pct", "Max_True_Act", "Top True Word", "Max_DblNeg_Act", "Top_DblNeg_Word"])

#Loop Over Neurons in Layer
        for n in target_neuron:
            records, true_exc, dbl_neg, top_10, inv_count = analyze_neuron(model, tokenizer, prompts, target_layer, n)
            print(f"Working on {n}")
            max_true_act = round(true_exc[0][0], 4) if true_exc else 0.0
            top_true_word = true_exc[0][3] if true_exc else "None"
            max_dbl_act = round(dbl_neg[0][0], 4) if dbl_neg else 0.0
            top_dbl_word = dbl_neg[0][3] if dbl_neg else "None"
            writer.writerow([
                target_layer, n, inv_count, inv_count*10,
                max_true_act, top_true_word, max_dbl_act, top_dbl_word])

if __name__ == "__main__":
    main()

