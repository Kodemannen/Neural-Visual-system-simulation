# Simulation of the visual system of a biological brain

## Abstract

A computationally effective toy model of the visual system of a biological brain, that can easily be extended to add more realism. The model takes images as input – representing visual stimuli from the eye – and outputs an estimation of the cortical LFP (local field potential) that is generated as cortex processes the input. 

Visual cortex is modeled as a set of recurrent and interconnected populations of excitatory and inhibitory spiking neurons (LIF neurons) using pyNEST, see https://nest-simulator.readthedocs.io/en/stable/index.html.

The activity of the cortical populations are driven by incomming spike trains that represents stimulus from LGN. The spike rate profile of these spike trains are calculated as a function of the input images through the pyLGN package, see https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006156.

The cortical LFP signals are estimated from the firing activity of the LGN and cortical neurons using hybridLFPy, see https://academic.oup.com/cercor/article/26/12/4461/2333943.

Summary: This Python package essentially simulates an eye that receives optical stimulus and sends the information to the visual cortex where an electrode is recording the electric field generated from the resulting neural activity.
