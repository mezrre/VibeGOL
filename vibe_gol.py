import os
import time
import random
import json


import numpy as np
import torch
import torch.nn.functional as F

import pygame

from vibe_core import GoLReverseNet


with open("config/vibe_gol_config.json", 'rt') as f:
    data = json.loads(f.read())

# Path to model file containing the Vibe GOL model
MODEL_PATH = data["model_path"]
# Size of the grid for the editor
GRID = int(data["grid_size"])

# Size of each grid cell in pixels
CELL_SIZE = 15

# Space between grid cells on screen
GRID_GAP = 2

# Width and height of a single NxN panel displaying a grid
PANEL_WIDTH = GRID * (CELL_SIZE + GRID_GAP) + 20
PANEL_HEIGHT = GRID * (CELL_SIZE + GRID_GAP) + 55

# Window dimensions
WINDOW_WIDTH = PANEL_WIDTH * 3 + 30
WINDOW_HEIGHT = PANEL_HEIGHT + 80

# Device to run model inferences on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def random_grid():
    return (np.random.random((GRID, GRID)) < 0.25).astype(np.uint8)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = GoLReverseNet(channels=ckpt["channels"], n_blocks=ckpt["n_blocks"]).to(DEVICE).eval()
    model.load_state_dict(ckpt["model_state"])
    model.half()

    print(f"Loaded {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Model: channels={ckpt['channels']}, blocks={ckpt['n_blocks']}")

    return model

def gol_step_numpy(grid):
    neighbors = np.zeros_like(grid, dtype=np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbors += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    born = (neighbors == 3) & (grid == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    return (born | survive).astype(np.uint8)

def gol_step_tensor(grid):
    kernel = torch.ones((1, 1, 3, 3), device=grid.device, dtype=grid.dtype)
    kernel[0, 0, 1, 1] = 0
    padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
    neighbors = F.conv2d(padded, kernel)[0, 0]
    born = (neighbors == 3) & (grid == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    return (born | survive).to(grid.dtype)

# generate a run length encoded GOL string from a binary numpy grid
def gen_rle(grid):
    cur_run = 0
    cur_value = 0
    string = ""
    value_names = ['b', 'o']
    
    for y in range(GRID):
        cur_value = grid[y, 0]
        for x in range(GRID):
            cur_run += 1
            if grid[y, x] != cur_value:
                string = string + (str(cur_run) if cur_run > 1 else "") + value_names[cur_value]
                cur_run = 0
                cur_value = grid[y, x]

        if cur_run > 0:
            string = string + (str(cur_run) if cur_run > 1 else "") + value_names[cur_value] + "$"
            cur_run = 0

    string = f"x = {GRID}, y = {GRID}, rule = B3/S23\n" + string[:-1] + "!"

    return string
            



# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# model_inference() allows direct access to the models inputs and outputs
# Parameters
#  model                 pytorch model
#  input_grid             NxN binary matrix: the grid state the model is trying to find the predecessor to
#  mask                  NxN binary matrix: mask with a '1' at each cell that known_values_grid has a known value
#  known_values_grid     NxN binary matrix: grid NxN containing any cells that are known (this is useful for discerning outputs when there are multiple valid predecessors, the model will want to output an average of them, resulting in many cells at probabilities between 0 and 1. A good way to get around this is to choose one of those cells and set it to alive or dead and add it to the mask, which makes the model then update the probabilbies on the others based on the added info)
#  prob_threshold        the minimum model probability for a cell to be predicted as alive
#  device                device for pytorch to use
# 
# Returns
#  probability           NxN matrix containing probability each cell is alive in the predecessor
#  prediction            NxN binary matrix where each cell is 1 if the probability is greater than `prob_threshold``
#  confidence            NxN binary matrix for how far the probabilities are from 0.5. Values close to 1 indicate high probability for alive or dead, while values close to 0 indicate equal probability for alive and dead.
@torch.no_grad()
def model_inference(model, input_grid, mask=None, known_values_grid=None, prob_threshold=0.5, device=None):
    input_grid = input_grid.astype(np.float32)

    # No predecessor information is provided, use blank mask and predecessor grid
    if mask is None or known_values_grid is None:
        mask = np.zeros_like(input_grid)
        known_values_grid = np.zeros_like(input_grid)

    # apply mask
    masked_prev = mask * known_values_grid

    # run model inference
    x = np.stack([input_grid, masked_prev, mask], axis=0)
    x = torch.from_numpy(x).unsqueeze(0).to(device, dtype=torch.float16)

    # get statistics for the mode
    probability = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    prediction = (probability >= prob_threshold).astype(np.uint8)
    confidence = np.abs(probability - 0.5) * 2.0

    return probability, prediction, confidence

# predict_incremental() is a smarter algorithm which uses the model to slowly build a solution
# it starts by taking the input grid and getting the probability matrix from the model with no information about the solutions in the mask. It then only gives the output cells with a high confidence with the mask, giving a new output with more high confidence cells. This is repeated multiple times until the full grid is given in the mask, meaning it is finished. It also runs GoL on each step will return if a model prediction if it matches the input_grid, meaning it has a solution.
# Parameters
#  model            pytorch model
#  input_grid       NxN binary grid of the GoL state to find a predecessor of
#  prob_threshold   probability required at model output for a cell to be considered alive
#  max_steps        Maximum number of steps to try revealing cells, after this many steps, it just returns what it has, even if it isnt a solution
#  device           device for pytorch to use
#
# Returns
#  probability      NxN binary grid containing probability each cell is alive
#  prediction       NxN binary grid containing the predicted previous state for the input
#  is_solution      boolean True if the prediction is a solution. If it reaches max_steps and the prediction is not a full solution, this is False.

@torch.no_grad()
def predict_incremental(model, input_grid, max_steps=1000, prob_threshold=0.98, device=DEVICE):
    if not torch.is_tensor(input_grid):
        input_grid = torch.from_numpy(input_grid)
    input_grid = input_grid.to(device=device, dtype=torch.float16)
    prediction = torch.zeros_like(input_grid)
    # mask = torch.zeros_like(input_grid)
    probability = torch.full_like(input_grid, 0.5)

    for _ in range(max_steps):
        x = torch.stack((input_grid, prediction, prediction), dim=0).unsqueeze(0)
        probability = torch.sigmoid(model(x))[0, 0]

        prediction = (probability >= prob_threshold).to(input_grid.dtype)
        
        input_grid_roundtrip = gol_step_tensor(prediction)
        score = torch.count_nonzero(input_grid_roundtrip != input_grid)
        if score == 0:
            break

        uncertainty = torch.abs(probability - prediction)

        # choose random cell in the top 3 probabilities not already revealed
        k = min(3, uncertainty.numel())
        uncertainties, indices = torch.topk(uncertainty.flatten(), k=k)
        idx = random.choices(indices.cpu().numpy(), uncertainties.cpu().numpy())

        # cells with lower probabilities are less likely to get flipped
        if torch.rand((), device=device) > uncertainty.flatten()[idx]:
            continue

        # flip the uncertain value, making it more certain.
        # alive cells are masked, so this adds it to the mask if it is set alive
        prediction.flatten()[idx] = 1 - prediction.flatten()[idx]

    is_solution = score == 0
    return probability.float().cpu().numpy(), prediction.to(torch.uint8).cpu().numpy(), is_solution

# ---------------------------------------------------------------------------
# Drawing and User Interaction
# ---------------------------------------------------------------------------

# draw a rectangle for a square in a cell
def cell_rect(panel_x, panel_y, row, col):
    x = panel_x + 10 + col * (CELL_SIZE + GRID_GAP)
    y = panel_y + 45 + row * (CELL_SIZE + GRID_GAP)
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

# get a color from probability 0-1
def probability_color(p):
    p = float(np.clip(p, 0, 1))

    if p < 0.5:
        t = p * 2
        return int(20 + 70 * t), int(30 + 90 * t), int(80 + 140 * t)

    t = (p - 0.5) * 2
    return int(90 + 165 * t), int(120 + 100 * t), int(220 - 180 * t)

# draw a grid panel to display a grid
def draw_grid(screen, panel_x, panel_y, grid, title, mode, grid_2=None):
    font = pygame.font.Font(None, 25)
    title_surface = font.render(title, True, (235, 235, 235))
    screen.blit(title_surface, (panel_x + 10, panel_y + 10))

    for row in range(GRID):
        for col in range(GRID):
            rect = cell_rect(panel_x, panel_y, row, col)
            value = float(grid[row, col])
            
            color = (255, 0, 255) # if cells are magenta, something went wrong

            if mode == "binary":
                color = (245, 245, 245) if value > 0.5 else (25, 25, 25)
            if mode == "binary-double":
                value2 = float(grid_2[row, col])
                if value > 0.5 and value2 > 0.5:
                    color = (245, 245, 245)
                elif value > 0.5 and value2 < 0.5:
                    color = (245, 245, 25)
                elif value < 0.5 and value2 > 0.5:
                    color = (245, 25, 25)
                else:
                    color = (25, 25, 25)

            if mode == "probability":
                color = probability_color(value)

            pygame.draw.rect(screen, color, rect)

# get the cell coordinate in a panel that the mouse is hovering over
# returns None when the mouse is outside the given panel
def mouse_to_cell(mouse_x, mouse_y, panel_x, panel_y):
    local_x = mouse_x - panel_x - 10
    local_y = mouse_y - panel_y - 45
    step = CELL_SIZE + GRID_GAP

    if local_x < 0 or local_y < 0:
        return None

    col = local_x // step
    row = local_y // step

    if not (0 <= row < GRID and 0 <= col < GRID):
        return None

    if local_x % step >= CELL_SIZE or local_y % step >= CELL_SIZE:
        return None

    return int(row), int(col)

# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

# Initiate pygame and store grids on screen
def main():
    # Pygame initialization
    pygame.init()
    pygame.display.set_caption("VibeGOL - Vibe Coding but for Conway's Game of Life")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # load the VibeGOL model
    model = load_model()


    # a set of numpy arrays for grids

    # This is the pattern you want to find the predecessor for
    input_grid = np.zeros((GRID, GRID), dtype=np.uint8)

    # Model probabilities (for each cell being alive) for previous grid state
    probability = np.full((GRID, GRID), 0.5, dtype=np.float32)

    # Model prediction for previous grid state
    prediction = np.zeros((GRID, GRID), dtype=np.uint8)

    # GOL step on prediction - rendered ontop of the input grid, mismatched cells highlighted
    roundtrip = np.zeros((GRID, GRID), dtype=np.uint8)

    # x values for each panel
    panel_x = [0, PANEL_WIDTH, PANEL_WIDTH * 2]
    # y value for all panels (they are rendered in a row)
    panel_y = 10

    mouse_button = None
    running = True

    # the square the user is hovering over
    hover_loc = (0, 0)    

    while running:
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_ESCAPE: # Press ESC to quit
                    running = False
                elif event.key == pygame.K_u: # Press U to step the input grid forward using GOL
                    input_grid = gol_step_numpy(input_grid)

                elif event.key == pygame.K_e: # Press E to set the input grid to the current prediction
                    input_grid = np.copy(prediction)
                    dirty = False
                elif event.key == pygame.K_c: # Press C to reset the input grid to 0
                    input_grid.fill(0)
                    dirty = True
                elif event.key == pygame.K_r: # Press R to initialize a random input grid
                    input_grid = random_grid()
                    dirty = True
                elif event.key == pygame.K_p: # Press P to use the model to predict
                    probability, prediction, _ = predict_incremental(model, input_grid)
                    roundtrip = gol_step_numpy(prediction)
                elif event.key == pygame.K_s: # Press S to save the current input grid to disk
                    rle_string = gen_rle(input_grid) 
                    with open("saved.rle", "wt") as f:
                        f.write(rle_string)
                    pass


            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_button = event.button
                row_col = mouse_to_cell(*event.pos, panel_x[0], panel_y)

                if row_col is not None:
                    row, col = row_col

                    if event.button == 1:
                        input_grid[row, col] = 1
                    elif event.button == 3:
                        input_grid[row, col] = 0
                    elif event.button == 2:
                        input_grid.fill(0)

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_button = None

            elif event.type == pygame.MOUSEMOTION and mouse_button in (1, 3):
                row_col = mouse_to_cell(*event.pos, panel_x[0], panel_y)

                if row_col is not None:
                    row, col = row_col
                    input_grid[row, col] = 1 if mouse_button == 1 else 0
                
                


        screen.fill((10, 10, 10))

        draw_grid(screen, panel_x[0], panel_y, input_grid, "Painted NEXT", "binary-double", grid_2=roundtrip)
        draw_grid(screen, panel_x[1], panel_y, prediction, "Predicted PREVIOUS", "binary")
        draw_grid(screen, panel_x[2], panel_y, probability, "Confidence", "probability")

        pygame.draw.circle(screen, (0, 255, 0), hover_loc, 2)

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
