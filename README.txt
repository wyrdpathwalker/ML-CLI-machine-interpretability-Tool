SwiGlu Sign Flip Checker
Proof of concept tool for identifying where positive neural activations are the result of a double-negative input, where both gate and up are less than zero, as opposed to a 'true' activation where gate and up are greater than zero. My understanding is that standard activation sorting does not differntiate between these types of activation.

This is a Python script that hooks into a target MLP and writes a sorted comparison table to a records folder.
It accepts the optional argument --prompt (defaults to internal test corpus), --layer, and --neuron as command line arguments.

Inspired by Nick Cammarata’s essay "Neural Networks Are Already Clean".

Written by Matthew Cinquina
