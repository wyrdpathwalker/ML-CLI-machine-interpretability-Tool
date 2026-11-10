import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from modelanalyze import loadmodel analyze_neuron

def main():
    parser = argparse.ArgumentParser(description="Inspect SwiGLU sign splits")
    parser.add_argument ("--prompt", type=str, default=None, help="Enter a single custom prompt to test")
    parser.add_argument("--neuron", type=int, default=None, help="Pick a specific neuron. Default will run a loop over all neurons!")
    parser.add_argument("--layer", type=int, default=15, help="Pick the transormer layer you'd like to test, default is 15")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135m", help="Enter a specific model to use. Default is SmolLM2-135m")
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

    if args.neuron:
        target_neuron = [args.neuron]
    else:
        target_neuron = #Loop over all neurons? Need to think about this code here

        print(f"Howdy, welcome. Calling model and running checks. Gathering results for Layer {target_layer}.")

#Call The model here, once.

#Call the analzer here, as many times as needed?

#Print Results
    if args.prompt:
        prompt_label = prompts[0].split(maxsplit=1))[0].strip("!?,.:;")
    else:
        prompt_label = "DefaultCorpus"

    if args.neuron:
        neuron_label = [args.neuron]
    else:
        neuron_label = "All"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("Records", exist_ok=True)
    filename = f"Records/{prompt_label}_L{target_layer}_N{neuron_label}_TestDateTime{timestamp}.csv"

    #Better to format large scale output as comma sepearated values than as txt I think
    #Will need to consult the machines, this one is new for me



