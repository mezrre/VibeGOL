"""
Train a CNN to reverse one step of Conway's Game of Life.

Input:
  Channel 0: 32x32 binary grid = the result AFTER one GoL step (the "next" state)
  Channel 1: partially revealed predecessor, with random cells masked to zero
  Channel 2: mask indicating which predecessor cells are actually revealed

Output:
  32x32 binary grid = the complete predecessor state.

The partial predecessor is supplied during training as additional information.
Because the inverse problem is non-unique, this lets the network condition its
prediction on a particular valid predecessor rather than independently mixing
cells from multiple possible predecessors.

The model is trained against the complete predecessor, including the revealed
cells. At inference time, the amount of predecessor information can be varied
by changing the mask probability.

We score:
  1. cell-wise accuracy against the specific predecessor used to build the pair
  2. exact predecessor match rate
  3. round-trip accuracy: step the predicted predecessor through real GoL rules
     and compare it to the supplied next state
  4. exact round-trip match rate
  5. consistency with the revealed predecessor cells
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import random
import matplotlib.pyplot as plt

from vibe_core import GoLReverseNet

torch.manual_seed(67)
np.random.seed(6767)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# grid size for training
# NOTE that this does not effect the grid size for inference, but
# small grid sizes may prevent the model from learning relationships
# over larger distances


# ---------------------------------------------------------------------------
# Game of Life mechanics (toroidal / wrap-around boundary, vectorized numpy)
# ---------------------------------------------------------------------------
def gol_step_numpy(grid):
    """grid: (N, GRID, GRID) uint8/bool array of 0/1. Returns next-state array."""
    neighbors = np.zeros_like(grid, dtype=np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbors += np.roll(np.roll(grid, dy, axis=1), dx, axis=2)
    born = (neighbors == 3) & (grid == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    return (born | survive).astype(np.uint8)

def gol_step_torch(grid):
    """grid: (N,1,GRID,GRID) float tensor of 0/1."""
    kernel = torch.ones((1, 1, 3, 3), device=grid.device, dtype=grid.dtype)
    kernel[0, 0, 1, 1] = 0
    padded = F.pad(grid, (1, 1, 1, 1), mode="circular")
    neighbor_count = F.conv2d(padded, kernel)
    born = (neighbor_count == 3) & (grid == 0)
    survive = ((neighbor_count == 2) | (neighbor_count == 3)) & (grid == 1)
    return (born | survive).float()


# ---------------------------------------------------------------------------
# Data generation - 
# ---------------------------------------------------------------------------
# Initialize a grid randomly with a set densities and step them forward with GoL
# training data are pairs of grids at consecutive steps
def make_dataset(n_samples, grid_size, seed=None, mask_probability_range=(0.1, 0.9)):
    """Generates random (prev, next) pairs with independently masked predecessors.

    Each sample gets its own mask probability drawn uniformly from
    mask_probability_range.
    """
    rng = np.random.default_rng(seed)

    prev = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8)
    densities = rng.uniform(0.15, 0.45, size=n_samples)

    for i, d in enumerate(densities):
        prev[i] = (rng.random((grid_size, grid_size)) < d).astype(np.uint8)

    nxt = prev
    for i in range(int(random.randint(6, 20))):
        prev = nxt
        nxt = gol_step_numpy(prev)

    # One mask probability per sample.
    mask_probabilities = rng.uniform(mask_probability_range[0], mask_probability_range[1], size=n_samples)

    # Broadcast each sample's probability across its whole grid.
    mask = (rng.random((n_samples, grid_size, grid_size)) >= mask_probabilities[:, None, None]).astype(np.uint8)

    masked_prev = prev * mask
    
    return prev, nxt, masked_prev, mask


# ChatGPT vomitted out a function to make data on the GPU
def make_batch_torch(batch_size, grid_size, device, mask_probability_range=(0.0, 1.0)):
    densities = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.25, 0.65)
    prev = (torch.rand(batch_size, 1, grid_size, grid_size, device=device) < densities).float()

    for _ in range(random.randint(6, 20)):
        prev = gol_step_torch(prev)

    nxt = gol_step_torch(prev)
    mask_prob = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*mask_probability_range)
    mask = (torch.rand_like(prev) >= mask_prob).float()
    masked_prev = prev * mask
    x = torch.cat((nxt, masked_prev, mask), dim=1)
    return x, prev

# streaming dataset that uses make_dataset() to create infnite unique training data
# the model will never overfit
class StreamingGoLDataset(torch.utils.data.IterableDataset):
    """Generates fresh random (next, masked_prev, mask, prev) batches."""

    def __init__(self, batch_size, batches_per_epoch, grid_size, seed=None, mask_probability_range=(0.1, 0.9)):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.grid_size = grid_size
        self.seed = seed
        self.mask_probability_range = mask_probability_range

    def __iter__(self):
        rng = np.random.default_rng(self.seed)

        for _ in range(self.batches_per_epoch):
            prev, nxt, masked_prev, mask = make_dataset(
                self.batch_size,
                self.grid_size,
                seed=rng.integers(1 << 30),
                mask_probability_range=self.mask_probability_range,
            )

            # Input has three channels:
            #   0 = next state
            #   1 = partially revealed predecessor
            #   2 = revelation mask
            x = torch.from_numpy(
                np.stack([nxt, masked_prev, mask], axis=1)
            ).float()

            # Complete predecessor is the target.
            y = torch.from_numpy(prev).float().unsqueeze(1)

            yield x, y, torch.from_numpy(mask).float().unsqueeze(1)





# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def evaluate(model, n_eval=2000, batch_size=200, seed=67, grid_size=32, mask_probability_range=(0, 1)):
    model.eval()

    rng = np.random.default_rng(seed)

    cellwise_correct = 0
    cellwise_total = 0
    exact_prev_matches = 0

    roundtrip_correct = 0
    roundtrip_total = 0
    exact_roundtrip_matches = 0

    revealed_correct = 0
    revealed_total = 0

    n_done = 0

    with torch.no_grad():
        while n_done < n_eval:
            bs = min(batch_size, n_eval - n_done)

            prev, nxt, masked_prev, mask = make_dataset(
                bs,
                grid_size=grid_size,
                seed=rng.integers(1 << 30),
                mask_probability_range=mask_probability_range,
            )

            x = torch.from_numpy(
                np.stack([nxt, masked_prev, mask], axis=1)
            ).float().to(DEVICE)

            y = torch.from_numpy(prev).float().unsqueeze(1).to(DEVICE)
            mask_t = torch.from_numpy(mask).float().unsqueeze(1).to(DEVICE)

            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()

            cellwise_correct += (pred == y).sum().item()
            cellwise_total += y.numel()

            exact_prev_matches += (
                (pred == y).all(dim=(1, 2, 3)).sum().item()
            )

            # The model should reproduce every predecessor cell that was
            # explicitly revealed to it.
            revealed_correct += (
                ((pred == y) * mask_t).sum().item()
            )
            revealed_total += mask_t.sum().item()

            # Round-trip through the actual GoL rule.
            forwarded = gol_step_torch(pred)

            x_next = x[:, 0:1]

            roundtrip_correct += (
                (forwarded == x_next).sum().item()
            )
            roundtrip_total += x_next.numel()

            exact_roundtrip_matches += (
                (forwarded == x_next).all(dim=(1, 2, 3)).sum().item()
            )

            n_done += bs

    model.train()

    return {
        "cellwise_acc": cellwise_correct / cellwise_total,
        "exact_prev_match_rate": exact_prev_matches / n_done,
        "roundtrip_cellwise_acc": roundtrip_correct / roundtrip_total,
        "roundtrip_exact_match_rate": exact_roundtrip_matches / n_done,
        "revealed_cell_acc": revealed_correct / max(revealed_total, 1),
        "n_eval": n_done,
    }


# Train the model -- black magic wizardry happens here
def train(
    epochs,
    batch_size,
    batches_per_epoch,
    grid_size,
    lr=1e-2,
    channels=64,
    n_blocks=8,
    mask_probability_range=(0, 1),
    save_path=None
):
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models/gol_reverse_model.pt",
        )

    model = GoLReverseNet(
        channels=channels,
        n_blocks=n_blocks,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    print(
        f"Model params: {n_params:,}  device: {DEVICE}",
        flush=True,
    )

    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 1

    last_epoch_done = start_epoch - 1
    total_training_data = 0
    
    loss_values = []
    roundtrip_acc_values = []
    exact_match_values = []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # ds = StreamingGoLDataset(
        #     batch_size,
        #     batches_per_epoch,
        #     seed=epoch * 67 + 6767,
        #     grid_size=grid_size,
        #     mask_probability_range=mask_probability_range,
        # )

        total_training_data += batch_size * batches_per_epoch * grid_size * grid_size
        # loader = torch.utils.data.DataLoader(
        #     ds,
        #     batch_size=None,
        # )

        running_loss = 0.0
        n_batches = 0


        for _ in range(batches_per_epoch):
            x, y = make_batch_torch(batch_size, grid_size, DEVICE, mask_probability_range=(0, 0.5))

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)

                # Train against the COMPLETE predecessor, not just the masked
                # cells. This encourages the output to represent one coherent
                # predecessor rather than only copying the known locations.
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            metrics = evaluate(
                model,
                n_eval=10,
                grid_size=grid_size,
                mask_probability_range=mask_probability_range,
            )

            # loss.backward()
            # opt.step()
            batch_loss = loss.item()
            running_loss += batch_loss
            n_batches += 1

            loss_values += [batch_loss]
            roundtrip_acc_values += [metrics['roundtrip_cellwise_acc']]
            exact_match_values += [metrics['exact_prev_match_rate']]

        sched.step()

        dt = time.time() - t0

        metrics = evaluate(
            model,
            n_eval=300,
            grid_size=grid_size,
            mask_probability_range=mask_probability_range,
        )

        # training size in bits
        total_training_size = epochs*batch_size*batches_per_epoch*grid_size*grid_size
        print(
            f"epoch {epoch:3d}/{epochs}  loss={running_loss / n_batches:.4f}  "
            f"cellwise_acc={metrics['cellwise_acc']:.4f}  "
            f"exact_prev_match={metrics['exact_prev_match_rate']:.4f}  "
            f"roundtrip_cellwise={metrics['roundtrip_cellwise_acc']:.4f}  "
            f"roundtrip_exact={metrics['roundtrip_exact_match_rate']:.4f}  "
            f"revealed_acc={metrics['revealed_cell_acc']:.4f}  "
            f"total training grids so far={total_training_data/8000000:.2f}MB/({total_training_size/8000000:.2f}MB) "
            f"({dt:.1f}s)",
            flush=True,
        )

        last_epoch_done = epoch

        torch.save(
            {
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "channels": channels,
                "n_blocks": n_blocks,
                "mask_probability_range": mask_probability_range,
                "epoch": epoch,
            },
            save_path,
        )

    print(
        f"Saved model to {save_path} (through epoch {last_epoch_done})",
        flush=True,
    )

    print(
        "\nFinal evaluation on 3000 fresh random samples:",
        flush=True,
    )

    final_metrics = evaluate(
        model,
        n_eval=3000,
        mask_probability_range=mask_probability_range,
    )

    for k, v in final_metrics.items():
        print(
            f"  {k}: {v}",
            flush=True,
        )

    plt.plot(list(range(0, len(loss_values))), loss_values)
    plt.plot(list(range(0, len(exact_match_values))), exact_match_values)
    plt.plot(list(range(0, len(roundtrip_acc_values))), roundtrip_acc_values)

    plt.xlabel("Batch #")
    plt.ylabel("") 
    plt.legend(["loss", "exact match rate", "roundtrip match rate"])
    plt.show()
    return model


if __name__ == "__main__":
    import json

    with open("config/training_config.json", 'rt') as f:
        data = json.loads(f.read())

    # Path to model file containing the Vibe GOL model
    model_path = data["model_path"]
    # Size of the grid for the editor
    grid_size = int(data["grid_size"])
    
    channels = int(data["channel_count"])
    n_blocks = int(data["layer_count"]) // 2

    epochs = int(data["num_epochs"]) 
    batch_size = int(data["batch_size"]) 
    batches_per_epoch = int(data["batches_per_epoch"]) 
    learning_rate = float(data["learning_rate"])

    # batch_size = 16
    # grid_size = 16
    # epochs = 128
    # batches_per_epoch = 16
    
    training_size = epochs * batches_per_epoch * batch_size

    print()

    print(f"Training on {training_size} grid pairs with width {grid_size} ({grid_size * grid_size * training_size/8000000:.2f}MB) across {epochs} epochs (batch size is {batch_size}, {batches_per_epoch} batches per epoch)")
    train(
        epochs=epochs,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        channels=channels,
        n_blocks=n_blocks,
        grid_size=grid_size,
        mask_probability_range=(0, 1),
        save_path=model_path,
        lr=learning_rate
    )