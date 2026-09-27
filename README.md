# VibeGOL

**VibeGOL** uses a convolutional neural network (CNN) to work backwards through **Conway's Game of Life**.

Given a Game of Life grid, VibeGOL attempts to predict a grid from the previous generation that evolves into the supplied state. Because the Game of Life is generally **non-invertible**—multiple predecessor states can produce the same state—the model uses an iterative inference strategy to progressively constrain its prediction until it finds a predecessor that reproduces the input.

The project includes both a PyTorch training pipeline and a small Pygame interface for experimenting with reverse Game of Life interactively.

## How It Works

A normal Game of Life simulation maps:

```text
previous state ──► Game of Life ──► next state
```

VibeGOL attempts to approximate the inverse:

```text
next state ──► neural network ──► predicted previous state
```

The neural network is a residual convolutional network that uses circular padding, matching the repository's implementation of Game of Life with wrap-around/toroidal boundaries.

The model takes three channels as input:

1. **Next state** — the Game of Life grid whose predecessor is being sought.
2. **Masked predecessor** — cells of a candidate predecessor that have already been revealed.
3. **Mask** — indicates which predecessor cells are known.

It outputs a single-channel probability map representing the probability that each cell in the predecessor was alive.

### Why the mask matters

Reverse Game of Life is not necessarily a one-to-one problem. A single state can have multiple valid predecessors.

Instead of asking the network to produce one deterministic answer with no additional information, VibeGOL can progressively reveal parts of a candidate predecessor and feed that information back into the model.

The inference process therefore looks approximately like:

```text
                 ┌──────────────────────────────┐
                 │                              │
                 ▼                              │
Input state ──► CNN ──► probability map ──► candidate predecessor
                 |                         + any additional constraints                           
                 ▼
     Run GOL and check if correct 
                 │
                 ▼
           Final Solution
```

The incremental predictor repeatedly evaluates the candidate, checks its Game of Life round trip, and modifies uncertain cells until it finds a matching predecessor or reaches its iteration limit.

## Features

* Reverse one generation of Conway's Game of Life using a CNN.
* Iterative predecessor prediction rather than relying on a single network output.
* Probability and confidence maps for model predictions.
* Round-trip verification using the actual Game of Life rules.
* Interactive Pygame grid editor.
* Random grid generation.
* Manual cell editing.
* Save grids as Run Length Encoded (`.rle`) patterns.
* Train the model entirely from procedurally generated Game of Life states.
* GPU-accelerated training through PyTorch/CUDA.
* Configurable inference grid size and model checkpoint.

## Repository Structure

```text
VibeGOL/
├── config/
│   └── vibe_gol_config.json
├── models/
│   └── gol_reverse_model_new.pt
├── train_model.py
├── vibe_core.py
├── vibe_gol.py
├── requirements.txt
├── saved.rle
└── LICENCE
```

### `vibe_core.py`

Contains the neural network architecture.

The model is `GoLReverseNet`, consisting of:

* A convolutional input stem.
* A configurable number of residual convolution blocks.
* A `1×1` convolutional output head.
* Circular padding throughout the network.

The default architecture is 64 channels with 24 residual blocks, although the checkpoint stores the architecture parameters used when the model was trained.

### `train_model.py`

Contains the training and evaluation pipeline.

Training data is generated procedurally rather than loaded from a fixed dataset. Random predecessor grids are generated, advanced through multiple Game of Life generations, and then partially masked before being presented to the network.

The training code evaluates several properties of the model:

* Cell-wise predecessor accuracy.
* Exact predecessor matches.
* Round-trip accuracy.
* Exact round-trip matches.
* Consistency with known predecessor cells.

### `vibe_gol.py`

Contains the interactive application and inference logic.

It loads the trained checkpoint, creates the Pygame interface, handles user input, performs prediction, and visualizes the model output. The application currently runs inference on the CPU.

### `config/vibe_gol_config.json`

Controls the inference grid and checkpoint location.

The current configuration is:

```json
{
  "grid_size": "24",
  "model_path": "models/gol_reverse_model_new.pt"
}
```

The inference application therefore expects the trained checkpoint at `models/gol_reverse_model_new.pt` unless this configuration is changed.

## Installation

Clone the repository:

```bash
git clone https://github.com/mezrre/VibeGOL.git
cd VibeGOL
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

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

* The input Game of Life state.
* The model's predicted predecessor.
* The resulting Game of Life round trip / probability visualization.

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
| **S**                   | Save the current grid as `saved.rle`                  |
| **Esc**                 | Quit                                                  |

These controls are implemented directly in the Pygame event loop.

## Game of Life Rules

VibeGOL uses the standard **B3/S23** Conway's Game of Life rules:

* A dead cell becomes alive with exactly 3 neighbors.
* A living cell survives with 2 or 3 neighbors.
* All other cells die or remain dead.

The implementation uses **circular/wrap-around boundaries**, meaning the top edge connects to the bottom edge and the left edge connects to the right edge. This applies to both the NumPy and PyTorch implementations.

## Training

The model can be trained using:

```bash
python train_model.py
```

Training data is generated on the fly.

A typical training example is constructed as follows:

```text
Random predecessor
       │
       ▼
Game of Life
       │
       ▼
Next state
       │
       ├───────────────► model input
       │
Predecessor
       │
       ▼
Randomly mask cells
       │
       ├───────────────► masked predecessor
       │
       └───────────────► predecessor mask
```

The three resulting input channels are concatenated and supplied to the network:

```text
Channel 0: next state
Channel 1: masked predecessor
Channel 2: predecessor mask
```

The target is the complete predecessor grid.

The training implementation can also generate batches directly on the GPU, allowing the synthetic dataset to be streamed without maintaining a large static dataset on disk.

### Important training detail

The training grid size and inference grid size are separate concepts in the implementation. The training code notes that training on larger grids can help the model learn relationships over larger spatial distances, while the inference application reads its grid size from `config/vibe_gol_config.json`.

## Inference

The lower-level inference function is:

```python
model_inference(model, input_grid, mask=None, known_values_grid=None, prob_threshold=0.5, device=None)
```

It returns:

1. A probability matrix.
2. A binary predecessor prediction.
3. A confidence matrix.

Confidence is calculated from the distance of each probability from `0.5`; values closer to `0` indicate uncertainty while values closer to `1` indicate stronger confidence.

For the interactive application, `predict_incremental()` provides a higher-level search strategy:

```python
predict_incremental(model, input_grid, max_steps=1000, prob_threshold=0.98, device=DEVICE)
```

At each iteration it:

1. Predicts the predecessor.
2. Applies the real Game of Life rules to that prediction.
3. Compares the resulting state with the requested input.
4. Stops if the round trip matches exactly.
5. Otherwise selects an uncertain cell and modifies the candidate.
6. Repeats until a solution is found or `max_steps` is reached.

This is important because the neural network itself is not being treated as a guaranteed inverse solver. The actual Game of Life simulation provides the final validity check.

## RLE Output

Pressing **S** writes the current input grid to:

```text
saved.rle
```

The generated file uses the standard B3/S23 rule declaration and Run Length Encoding to represent the grid.

For example, the output begins with metadata similar to:

```text
x = 24, y = 24, rule = B3/S23
```

This makes saved patterns convenient to use with other Conway's Game of Life tooling.

## Limitations

VibeGOL should be viewed as an experimental learned inverse rather than an exact mathematical solver.

### Non-unique predecessors

A Game of Life state can have multiple valid predecessors. The model therefore learns a distribution over plausible cells rather than having a uniquely determined answer in every case.

The iterative inference algorithm attempts to resolve this ambiguity by progressively modifying uncertain cells and checking the resulting candidate against the actual Game of Life transition function.

### No guaranteed solution

If the incremental search reaches its maximum number of iterations without finding a matching predecessor, it returns its current candidate and reports that it is not a solution.

### CPU inference

The interactive application currently sets its inference device to CPU. Training, on the other hand, is configured around CUDA in `train_model.py`.

### Model-dependent behavior

Prediction quality depends on the training configuration, model checkpoint, grid characteristics, and amount of predecessor information available to the network.

## Configuration

The primary inference configuration is:

```text
config/vibe_gol_config.json
```

```json
{
  "grid_size": "24",
  "model_path": "models/gol_reverse_model_new.pt"
}
```

### `grid_size`

Controls the size of the interactive Game of Life board.

### `model_path`

Specifies the PyTorch checkpoint loaded by `vibe_gol.py`.

If you train a new model and save it under a different filename, update `model_path` accordingly.

## Project Architecture

At a high level, the repository separates the system into three layers:

```text
┌─────────────────────────────────────────┐
│             Pygame Interface            │
│              vibe_gol.py                │
│                                         │
│  Grid editing / visualization / input   │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│             Inference Logic             │
│              vibe_gol.py                │
│                                         │
│  CNN prediction → candidate → roundtrip │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│              Neural Network             │
│              vibe_core.py               │
│                                         │
│       GoLReverseNet / ConvBlocks        │
└─────────────────────────────────────────┘
                     ▲
                     │
┌────────────────────┴────────────────────┐
│             Training Pipeline           │
│             train_model.py              │
│                                         │
│ Synthetic GoL data → masked inputs      │
│ → training → checkpoint                 │
└─────────────────────────────────────────┘
```

This separation makes `vibe_core.py` reusable independently of the interactive application, while `train_model.py` handles model development and `vibe_gol.py` handles the user-facing inference workflow.

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
