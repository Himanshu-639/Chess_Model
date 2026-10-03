# Supervised CNN Chess Playing Model

A production-grade, modular deep learning pipeline for training a supervised Residual Convolutional Neural Network (Dual-Head Policy + Value ResNet) to play chess.

---

## 🌟 Key Features

1. **Dual-Head Deep ResNet Architecture**:
   - **Shared Backbone**: Configurable Residual blocks (default: 6 blocks with 128 filters; scalable to 20 blocks with 256 filters).
   - **Policy Head**: Predicts probability distribution across all 16,384 discrete chess moves ($64 \times 64$ from-to squares + underpromotions).
   - **Value Head**: Predicts scalar evaluation $V \in [-1.0, 1.0]$ representing expected game outcome from the current player's perspective.
2. **Canonical 18-Plane Board Representation**:
   - Rotates the board so the active player is always White (`board.mirror()`).
   - Halves parameter space and doubles sample efficiency.
   - Encodes pieces, castling rights, en-passant square, and 50-move halfmove clock.
3. **100% Legal Move Masking**:
   - Applies dynamic illegal-move masking ($-\infty$ logit mask) during inference.
   - Guarantees 0% illegal move rate in real matches.
4. **Resilient Training with Anomaly Protection**:
   - Automated Mixed Precision (`torch.amp.autocast`) for fast GPU training.
   - Gradient clipping (`clip_grad_norm_`).
   - Real-time `RuntimeHealthMonitor` detecting NaNs, Infs, or diverging loss, triggering safe emergency checkpoint preservation before aborting.
   - Top-1 and Top-5 accuracy tracking, Value MAE, and automated CSV metric history.
5. **High-Performance Sharded Data Pipeline**:
   - Parses raw `.pgn` files into compressed `.npz` binary shards.
   - Supports multi-million game datasets without memory exhaustion.
6. **Unified Runner Script (`run_pipeline.py`)**:
   - Single entry point for verification, preprocessing, training, evaluation, and playing.

---

## 📁 Project Structure

```
Chess/
├── README.md               # Complete guide and step-by-step breakdown
├── config.yaml             # Hyperparameters, paths, and training configuration
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Master single-script pipeline entrypoint
├── src/
│   ├── __init__.py
│   ├── board_encoder.py    # 18-plane canonical board representation
│   ├── move_encoder.py     # Move to discrete index (16,384) & legal masking
│   ├── model.py            # Dual-Head ChessResNet model definition
│   ├── dataset.py          # Sharded NPZ dataset & synthetic PGN generator
│   ├── preprocess.py       # PGN parser and NPZ shard exporter
│   ├── trainer.py          # Resilient training loop & metric logging
│   ├── health_checks.py    # Pre-flight sanity tests & runtime anomaly monitor
│   ├── engine.py           # Chess AI inference engine with temperature sampling
│   └── evaluator.py        # Benchmark tournament against baseline bots
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Unit & integration test suite
└── data/                   # PGN and processed shards storage
```

---

## 🚀 Quickstart & Verification (Local Machine)

Verify that the entire codebase, forward/backward passes, data loader, trainer, and game evaluation work out of the box:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
python -m unittest discover tests

# 3. Run full verification self-test
python run_pipeline.py --mode verify
```

---

## 🖥️ Running on Target Training System (Full Dataset)

### Step 1: Transfer Code to Training Machine
Copy this `Chess/` directory to your target GPU machine (e.g. AWS, RunPod, Lambda Labs, Google Colab, or local GPU rig).

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Download PGN Dataset
You can use any standard chess PGN dataset. Recommended sources:
- **Lichess Elite Database** (games played by 2400+ / 2500+ Elo rated players):
  ```bash
  # Download a monthly archive (contains ~380,000 elite games / ~25M positions)
  wget https://database.nikonoel.fr/lichess_elite_2020-08.zip
  # Unzip archive into data/
  unzip lichess_elite_2020-08.zip -d data/
  ```
- **FIDE Grandmaster Games** or **KingBase / TWIC**.

### Step 3: Preprocess PGN into Fast NPZ Shards
Convert the PGN into compressed shards (e.g., 50,000 positions per shard):
```bash
python run_pipeline.py --mode preprocess --pgn data/lichess_elite.pgn --out data/processed --shard-size 50000 --val-split 0.05
```
*Expected Output:*
- `data/processed/train/shard_00000.npz`, `shard_00001.npz`, ...
- `data/processed/val/shard_00000.npz`, ...
- Preprocessing summary reporting total games and positions extracted.

### Step 4: Run Training on Full Dataset
Start training with automated GPU mixed precision, gradient clipping, metric logging, and anomaly checks:
```bash
python run_pipeline.py --mode train --config config.yaml --batch-size 256 --epochs 20
```

To resume training from a saved checkpoint:
```bash
python run_pipeline.py --mode train --config config.yaml --resume checkpoints/latest_model.pt
```

*Logged Outputs during Training:*
- Console updates showing Epoch, Train Loss, Train Policy Loss, Train Top-1 Acc, Train Top-5 Acc, Val Loss, Val Top-1 Acc, Val MAE, Learning Rate, and Elapsed Time.
- CSV metrics logged to `logs/training_history.csv`.
- Best model saved automatically to `checkpoints/best_model.pt`.
- Latest checkpoint saved to `checkpoints/latest_model.pt`.
- Anomaly safety: If loss exceeds threshold or produces NaN/Inf, an emergency state `checkpoints/diverged_checkpoint_epoch_X.pt` is saved and execution is safely halted.

### Step 5: Evaluate Trained Model
Benchmark the trained network in a tournament against baseline bots:
```bash
# Play 20 matches against Random bot
python run_pipeline.py --mode evaluate --checkpoint checkpoints/best_model.pt --games 20 --opponent random

# Play 20 matches against Greedy material bot
python run_pipeline.py --mode evaluate --checkpoint checkpoints/best_model.pt --games 20 --opponent greedy
```

### Step 6: Interactive Terminal Play
Play a game in your terminal against the trained model:
```bash
python run_pipeline.py --mode play --checkpoint checkpoints/best_model.pt --interactive
```

---

## ⚙️ Configuration Reference (`config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model.num_filters` | `128` | Number of convolution channels (use 256 for large models) |
| `model.num_blocks` | `6` | Number of ResNet residual blocks (scale to 10-20 for high Elo) |
| `training.batch_size` | `256` | Batch size (scale based on GPU VRAM) |
| `training.learning_rate` | `0.001` | Initial AdamW learning rate |
| `training.use_amp` | `true` | Enables FP16 mixed precision on CUDA |
| `training.clip_grad_norm` | `1.0` | Gradient norm clipping limit |
| `health_checks.max_loss_threshold` | `50.0` | Max allowable loss before stopping due to divergence |
