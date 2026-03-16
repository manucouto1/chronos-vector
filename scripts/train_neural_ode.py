#!/usr/bin/env python3
"""Train a Neural ODE model for ChronosVector trajectory prediction.

This script trains a GRU-ODE model that predicts future embedding vectors
from temporal trajectories. The trained model is exported as TorchScript
for inference in Rust via tch-rs (RFC-003).

Usage:
    pip install torch torchdiffeq
    python scripts/train_neural_ode.py \
        --dim 128 \
        --latent-dim 64 \
        --hidden-dim 128 \
        --epochs 100 \
        --output models/neural_ode_d128.pt

Model I/O Contract (must match cvx-analytics/src/torch_ode.rs):
    Input:
        trajectory: Tensor[1, T, D+1]  (normalized_time ++ vector)
        target_t:   Tensor[1, 1]        (always 1.0 = normalized target)
    Output:
        predicted:  Tensor[1, D]         (predicted vector at target_t)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ODEFunc(nn.Module):
    """Learned dynamics f(t, y) for dy/dt = f(t, y)."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def euler_integrate(
    func: ODEFunc, y0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, steps: int = 20
) -> torch.Tensor:
    """Simple Euler integration (TorchScript-compatible, no torchdiffeq needed)."""
    dt = (t1 - t0) / steps
    y = y0
    t = t0
    for _ in range(steps):
        y = y + dt * func(t, y)
        t = t + dt
    return y


class NeuralODEPredictor(nn.Module):
    """Encodes trajectory via GRU, integrates ODE, decodes to vector.

    Compatible with TorchScript export via torch.jit.script().
    Uses Euler integration instead of torchdiffeq for scriptability.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(input_dim + 1, latent_dim, batch_first=True)
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, trajectory: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: [B, T, D+1] — normalized time + vector per step
            target_t:   [B, 1]      — normalized target timestamp (typically 1.0)

        Returns:
            predicted:  [B, D]      — predicted vector at target time
        """
        # Encode trajectory into latent state
        _, h = self.encoder(trajectory)  # h: [1, B, latent]
        h = h.squeeze(0)  # [B, latent]

        # Last observed normalized time
        last_t = trajectory[:, -1, 0:1]  # [B, 1]

        # Integrate ODE from last observed time to target time
        predicted_latent = euler_integrate(
            self.ode_func,
            h,
            last_t.squeeze(-1).mean(),
            target_t.squeeze(-1).mean(),
            steps=20,
        )

        # Decode latent to vector space
        return self.decoder(predicted_latent)  # [B, D]


class SyntheticTrajectoryDataset(Dataset):
    """Generate synthetic trajectories for training.

    Creates random walk trajectories with optional regime changes,
    suitable for training the Neural ODE predictor.
    """

    def __init__(self, n_samples: int, dim: int, seq_len: int = 20, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.trajectories = []
        self.targets = []

        for _ in range(n_samples):
            # Random walk with drift
            drift = torch.randn(dim) * 0.01
            noise_scale = 0.05
            points = [torch.randn(dim)]
            for t in range(1, seq_len + 1):
                next_point = points[-1] + drift + torch.randn(dim) * noise_scale
                points.append(next_point)

            # Trajectory = first seq_len points, target = last point
            traj_vectors = torch.stack(points[:seq_len])  # [T, D]
            target_vector = points[seq_len]  # [D]

            # Add normalized timestamps: [0, 1/(T+1), 2/(T+1), ..., (T-1)/(T+1)]
            times = torch.linspace(0, (seq_len - 1) / seq_len, seq_len).unsqueeze(-1)  # [T, 1]
            traj_with_time = torch.cat([times, traj_vectors], dim=-1)  # [T, D+1]

            self.trajectories.append(traj_with_time)
            self.targets.append(target_vector)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.targets[idx]


def train(args):
    """Train the Neural ODE predictor."""
    print(f"Training Neural ODE: dim={args.dim}, latent={args.latent_dim}, "
          f"hidden={args.hidden_dim}, epochs={args.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = NeuralODEPredictor(
        input_dim=args.dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Synthetic training data
    dataset = SyntheticTrajectoryDataset(
        n_samples=args.n_samples, dim=args.dim, seq_len=args.seq_len
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Target time: always 1.0 (normalized)
    target_t = torch.ones(args.batch_size, 1, device=device)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        for traj_batch, target_batch in loader:
            traj_batch = traj_batch.to(device)
            target_batch = target_batch.to(device)
            bs = traj_batch.size(0)

            # Adjust target_t for last batch
            t_t = target_t[:bs]

            predicted = model(traj_batch, t_t)
            loss = criterion(predicted, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}, loss={avg_loss:.6f}")

    # Export as TorchScript
    model.eval()
    model = model.to("cpu")

    example_traj = torch.randn(1, args.seq_len, args.dim + 1)
    example_t = torch.ones(1, 1)

    scripted = torch.jit.trace(model, (example_traj, example_t))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    print(f"\nModel saved to {output_path}")
    print(f"  Input:  trajectory[1, T, {args.dim + 1}] + target_t[1, 1]")
    print(f"  Output: predicted[1, {args.dim}]")


def main():
    parser = argparse.ArgumentParser(description="Train Neural ODE for ChronosVector")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimensionality")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent ODE dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--seq-len", type=int, default=20, help="Trajectory sequence length")
    parser.add_argument("--n-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/neural_ode.pt", help="Output path")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
