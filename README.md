# VibeGOL

**VibeGOL** uses a convolutional neural network (CNN) to work backwards through **Conway's Game of Life**.

Given a Game of Life grid, VibeGOL attempts to predict a grid from the previous generation that evolves into the supplied state. Because the Game of Life is generally **non-invertible**—multiple predecessor states can produce the same state—the model uses an iterative inference strategy to progressively constrain its prediction until it finds a predecessor that reproduces the input.

The project includes both a PyTorch training pipeline and a small Pygame interface for experimenting with reverse Game of Life interactively.

## How It Works

Standard Conway's Game of Life (GOL) applies a deterministic rule to a grid of cells, which are alive or dead, giving the single valid next state. This is a computationally inexpensive operation that can be parallelized very easily, as each cells next state is just a direct combinational function for the cells current state and the 8 neighbor cells states. Reversing this algorithm, however, is much harder, as information is lost with each GOL step, meaning multiple states can result in the same result when applying GOL. Each cells previous state ends up depending on much more than the small 3x3 neighborhood and finding a previous grid requires search algorithms that require exponentially more compute for larger pattern sizes. VibeGOL attempts to use machine learning to optimize this search process by prioritizing states that are more likely to be predecessors than standard search algorithms. It uses a large CNN (Convolutional NN) to do this, which takes a result GOL state and map of constrained cells and finds a map of probabilities that each cell in the grid is alive in the previous state based on the input state and known information. 

The model takes three channels as input:
1. **Next state** — the Game of Life grid whose predecessor is being sought.
2. **Masked predecessor** — cells of a candidate predecessor that have already been revealed.
3. **Mask** — indicates which predecessor cells are known.

It outputs a single-channel probability map representing the probability that each cell in the predecessor was alive.

### Why the mask matters
As stated before, reversing GOL is not necessarily a one-to-one problem. A single state can have multiple valid predecessors and some states may have no predecessors (these are called Garden of Eden states, as they can only occur as the initialized state of a GOL run). Instead of asking the network to produce one deterministic answer with no additional information, VibeGOL can be used with a searching algorithm that progressively reveal parts of a candidate predecessor and feeds that information back into the model, until it finds a valid predecessor state. Finding a solutions can take a lot of compute, but verifying a potential solution can be done very quickly, as it just requires a forward GOL step and comparison. The incremental predictor implemented in vibe_gol.py repeatedly evaluates the candidate, checks its Game of Life round trip, and modifies uncertain cells until it finds a matching predecessor or reaches its iteration limit. This limit is necessary in the case of Garden of Eden states and when a poor model is used that never finds the solution.

### `vibe_core.py`

Contains the neural network architecture.
The model is `GoLReverseNet`, consisting of:
* A convolutional input stem.
* A configurable number of residual convolution blocks.
* Circular padding throughout the network with wrap-around (so all input states are assumed to be toriodal).

The default architecture is 16 channels with 5 residual blocks, although the checkpoint stores the architecture parameters used when the model was trained, so it can run any model size without changing any code. The largest model tested was 24 residual blocks with 64 channels, which showed early success.

### `train_model.py`
Contains the training and evaluation pipeline.
Training data is generated procedurally rather than loaded from a fixed dataset. Random predecessor grids are generated, advanced through multiple Game of Life generations, and then partially masked before being presented to the network, which is computationally cheap when parallelized.

The training code evaluates several properties of the model:
* Cell-wise predecessor accuracy.
* Exact predecessor matches.
* Round-trip accuracy.
* Exact round-trip matches.
* Consistency with known predecessor cells.
However, training loss is just the error between the particular predecessor state in the data and model prediction.

### `vibe_gol.py`
Contains the interactive application and inference logic.
It loads the trained checkpoint, performs predictions, and visualizes the model output. It requires Pygame to work.

### `config/vibe_gol_config.json`
The current configuration is:
```json
{
  "grid_size": "24",
  "model_path": "models/your_model.pt"
}
```
It stores the inference grid size and checkpoint file path. The inference application therefore expects the trained checkpoint at `models/your_model.pt` unless this configuration is changed.

## Installation
### Clone the repository:
```bash
git clone https://github.com/mezrre/VibeGOL.git
cd VibeGOL
```

### Create a virtual environment:
```bash
python -m venv .venv
```

### Activate it 
on Linux/macOS:
```bash
source .venv/bin/activate
```

on Windows:
```powershell
.venv\Scripts\activate
```

### Install the dependencies:
```bash
pip install -r requirements.txt
```

The repository currently specifies NumPy, Matplotlib, Pygame, PyTorch, and TorchVision. The PyTorch dependencies in `requirements.txt` target a CUDA 13.2 build, so users without a compatible CUDA environment may need to install an appropriate PyTorch build separately.

## Running VibeGOL
Once the model checkpoint exists at the configured path, launch the interactive application with:
```bash
python vibe_gol.py
```

The application opens a Pygame window containing three views of the grid:
* The input Game of Life state. (Left, this is the one you edit)
* The model's predicted predecessor. (Middle)
* The resulting Game of Life round trip / probability visualization. (Right)

The model is loaded from the path specified by `config/vibe_gol_config.json`.

## Controls

| Input                   | Action                                                |
| ----------------------- | ----------------------------------------------------- |
| **Left mouse button**   | Set a cell alive                                      |
| **Right mouse button**  | Set a cell dead                                       |
| **Middle mouse button** | Clear the input grid                                  |
| **R**                   | Generate a random input grid                          |
| **P**                   | Run predecessor prediction                            |
| **U**                   | Advance the input grid by one Game of Life generation |
| **E**                   | Replace the input with the current prediction         |
| **C**                   | Clear the input grid                                  |
| **S**                   | Save the current grid as `saved.rle` (can be imported to Golly)                 |
| **Esc**                 | Quit                                                  |

These controls are implemented directly in the Pygame event loop.

## Game of Life Rules

VibeGOL uses the standard **B3/S23** Conway's Game of Life rules:

* Cells with 3 neighbors are set alive (1)
* Cells with 2 neighbors keep their last state
* All other cells are set dead (0)

The implementation uses **circular/wrap-around boundaries**, meaning the top edge connects to the bottom edge and the left edge connects to the right edge. This applies to both the NumPy and PyTorch implementations.
It is not necessary to use the same grid size the model was trained on during inference, though performance may vary between vastly different sized training and inference grid sizes.
## Training

The model can be trained using:
```bash
python train_model.py
```

### Note: Model grid size is not hardcoded
The training grid size and inference grid size are separate concepts in the implementation. Training on larger grids can help the model learn relationships over larger spatial distances. The inference application reads its grid size from `config/vibe_gol_config.json`, which is separate from the training configuration.


## RLE Output (vibe_gol.py)
Pressing **S** writes the current input grid to:
```text
saved.rle
```

The generated file uses the standard B3/S23 rule declaration and Run Length Encoding to represent the grid. For example, the output begins with metadata similar to the following:
```text
x = 24, y = 24, rule = B3/S23
```

This makes saved patterns convenient to use with other Conway's Game of Life tooling such as Golly.

## Limitations
VibeGOL should be viewed as an experimental learned inverse rather than an exact mathematical solver. It is meant to be used in search algorithms that ensure the solution is correct, as the model can always output incorrect predictions. 

It outputs a probability map of each cell being alive, not a single output predecessor state. This is actually beneficial for some contexts, as you can find multiple predecessors and set constraints.

If the incremental search reaches its maximum number of iterations without finding a matching predecessor, it returns its current candidate and reports that it is not a solution. There may still be a solution it hasn't found, or it may be a Garden of Eden pattern.

Prediction quality depends on the training configuration, model checkpoint, grid characteristics, and amount of predecessor information available to the network.

## Development
The repository is intentionally small and self-contained. The primary files to modify are:
* **Model architecture:** `vibe_core.py`
* **Training/data generation:** `train_model.py`
* **Inference/search strategy:** `vibe_gol.py`
* **Interactive UI:** `vibe_gol.py`
* **Inference configuration:** `config/vibe_gol_config.json`

When experimenting with the model architecture, make sure the checkpoint's architecture parameters remain compatible with the model loader. The application reads the saved channel count and number of residual blocks from the checkpoint when reconstructing `GoLReverseNet`. 

## License
VibeGOL is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. See [`LICENCE`](./LICENCE) for the complete license text.

## Repository
[github.com/mezrre/VibeGOL](https://github.com/mezrre/VibeGOL)
