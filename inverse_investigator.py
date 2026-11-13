import os
import argparse
import csv
from datetime import datetime
from model_analyze import load_model, analyze_neuron

def main():
    parser = argparse.ArgumentParser(description="Inspect SwiGLU sign splits")
    parser.add_argument ("--prompt", type=str, default=None, help="Enter a single custom prompt to test")
    parser.add_argument("--neuron", type=int, default=None, help="Pick a specific neuron. Default will run a loop over all neurons!")
    parser.add_argument("--layer", type=int, default=None, help="Pick the transormer layer you'd like to test, default is all.")
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
        target_neuron = range(3)
    else:
        target_neuron = [args.neuron]

        #The Full Range for the default model is 1536 
    
    print(f"Commencing, Checking Args")

    model, tokenizer = load_model(args.model)
 
    if args.layer is None:
        target_layers = range(len(model.model.layers))
    else:
        target_layers = [args.layer]

    timestamp = datetime.now().strftime("%Y%m%d")
    os.makedirs("Records", exist_ok=True)
    filename = f"Records/Captures_{timestamp}.csv"
    file_exists = os.path.isfile(filename)

    print(f"Preparing record file for {timestamp}")

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Layer", "Neuron", "Top10_Inversions", "Inversion_Pct", 
                "Max_True_Act", "Top True Word", "Max_DblNeg_Act", "Top_DblNeg_Word"
                ])

        for target_layer in target_layers:
            layer_rows = analyze_neuron(model, tokenizer, prompts, target_layer, target_neuron)
            print(f"Getting results from the analyzer, writing")
            for row in layer_rows:
                writer.writerow(row)

if __name__ == "__main__":
    main()

