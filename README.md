# NeuralNetowrksTS
This is my first using TS in attempt to create a neural network. Well, I used a very strange 
algorithm (let's hope it works(in theory it should)) for backpropagation and layers don't exist, 
this module can consume a big chunk of your RAM.

The first version supports the most basic functions. One of them is backpropagation 
which works quite different from the other ones as it doesn't look far back, instead 
it asks it's neighbours (what's their value) and then the neuron adjusts it's weight 
based on that information. 

In theory this neural network should fix itself starting from the output "layer" to the new one.
So if it's like in the theory values in all of the neural network should drop at first and then 
the values (in absolute value) should increase after some iterations.

It's recommended to use only 4 main classes as it's not needed to use other ones.
Those classes are: TrainingExample, NeuralNetworkOptions, NeuralNetwork and 
NonAndLinearFunctions (this one contains all functions you could want to use).





