# Simulation of the visual system of a biological brain

## Abstract

We present a computationally effective toy model of the visual system of a
biological brain, that can easily be extended to add more realism. The model
takes images as input – representing visual stimuli from the eye – and outputs an
estimation of the cortical LFP (local field potential) that is generated as cortex
processes the input. We run a large number of simulations, each stimulated by a
randomized sequence of 10 images, and use the output data to train deep learning
algorithms (CNN and LSTM) to classify pieces of the LFP by input image.
The classifiers reach accuracies of 66 and 65%, averaged across all 10 inputs,
suggesting that the LFP indeed contain information about the stimulus that a
brain is processing. They are also more likely to confuse the LFPs of images that
qualitatively seem visually similar. We observe that a trained CNN transfers
better to test data that deviates slightly from the training set, but that the
LSTM seems marginally better at handling noise.

This package simulates a Brunel network (two populations of point neurons) that receives input from an LGN population, which again receives "optical" input from retinal ganglion cells. The Local Field Potential (LFP) is then calculated from the spiking activity using hybridLFPy.

The network essentially simulates an eye that sends information to the visual cortex where an electrode is recording the electric field generated from the resulting neural activity.

