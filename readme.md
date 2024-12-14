### Eon 0: Chaos
(Aug 18, 2024 - Nov 23, 2024)

#### Era 0: Blubzers
* pygame PoC
* Pre-monad creations

#### Era 1: Monadology
* ANN PoC
* Creations with a feed-forward action mechanism
	* Monad05: simple 5-neuron monad with no activation function (chemotaxis)
	* Monad15: simple 5-neuron monad with tanh

#### Era 2: Failing Abiogenesis
* pytorch PoC
* Creations with fully tensor-based movements
	* Monad211: 11-neuron monad with full IO flattening and LMWH sensory input
	* Monad313: 13-neuron monad with RNN
	* Monad429: 29-neuron monad with 2 recurrent layers governed by 316 parameters
							modulated by 632 genes, having an action organ to broadcast a
							message of tanh(float32) and a sensory organ to receive the
							message with its direction
  * Monad529: 5th gen monad with 29 neurons and 2 LSTM layers governed by 1156
							parameters modulated by 2312 genes
  * Monad5173: 5th gen monad with 5 LSTM layers each with 32 units, making a
							 total of 173 neurons, 38788 parameters and 77576 genes
  * Monad60T6x64: Monad with a memory organ of 4 tanh(float32) variables and
								  messaging capability of 2 tanh(float32) variables, controlled
								  by a transformer network of 6 layers, 16 dimensions, 4
								  attention heads, and 4x feed-forward expansion with bias term present LayerNorm and FF layers, governed by ~300k parameters

### Eon 1: Abundance
(Nov 23, 2024 - *ongoing*)

#### Era 0: 270 Balls
* nn2: Simple feed-forward network with 2 hidden layers using ReLU, where the
			 first hidden layer has 4x expansion from the input layer, and the second
			 hidden layer is either of the size of the input layer or 2x the size of
			 the output layer, whichever is larger, finally transformed into the
			 output layer with tanh
* Monads controlled by an nn2
	* Monad6105: Monad with 12 input, 5 output, 4 memory (both input & output),
							 and 80 hidden neurons, governed by 2281 parameters
	* Monad7A217: Monad with 16 input, 9 output, 16 memory, and 160 hidden
							  neurons, governed by 9177 parameters - LMWS and messaging
							  neurons are removed; ability to sense the resource gradient and
								to apply force on the resource grid are added
  * Monad7B105: Monad7A217 minus memory (down to 2k parameters; faster not
								step-wise but learning-wise)
  * Monad8A203: Monad with 32 input, 11 output, and 160 hidden neurons, governed
								by 8715 parameters, having the ability to sense and move
								surrounding structural units
  * Monad8B265: Monad with 32 input, 35 output, and 198 neurons, governed by
								15739 parameters, which can control the surrounding structural units' manipulation of resources
* KDTree to calculate sparse distance matrix
* conv2d to calculate resource diffusion

#### Era 1: Celestial Aquatics
* Toroidal boundaries
* Monad8C287: Same as Monad8B265, except having an internal state (works same as
						 	a memory organ - acts as both input and output) of 4 tanh(float32) variables (18527 parameters)



--\
.