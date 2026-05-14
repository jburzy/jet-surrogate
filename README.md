# jet-surrogate

Analysis pipeline for a jet surrogate model. Reads HEPMC events, reconstructs truth jets, associates truth particles, runs ONNX inference, and computes the two-jet acceptance.

## Pipeline

1. Read events from a HEPMC2/3 file
2. Reconstruct anti-kt R=0.4 truth jets from stable visible particles (ATLAS convention: excludes neutrinos and muons)
3. Recluster R=0.4 jets into anti-kt R=1.0 jets; require pT > 200 GeV
4. Associate all truth particles within ΔR < 1.4 of each large-R jet axis
5. Extract 32 features per particle (4 integer + 28 float, see [Features](#features))
6. Run ONNX surrogate model; assign a score to each jet
7. Cut at score > 0.5; report the fraction of events with ≥ 2 jets passing

## Installation

### 1. Create the environment

```bash
mamba create -n jet-surrogate python=3.11
mamba activate jet-surrogate
```

### 2. Install the C++ compiler (needed to build pyjet/FastJet)

```bash
mamba install -c conda-forge gxx_linux-64
```

### 3. Install the package

```bash
pip install -e /path/to/jet-surrogate
```

### GPU inference (optional)

By default `onnxruntime` (CPU) is installed. To use a CUDA GPU instead:

```bash
pip install onnxruntime-gpu
```

This replaces the CPU build; no code changes are needed.

## Usage

```
jet-surrogate HEPMC_FILE --model MODEL [OPTIONS]
```

### Required arguments

| Argument | Description |
|---|---|
| `HEPMC_FILE` | Path to input HEPMC2 or HEPMC3 file |
| `--model / -m` | Path to ONNX surrogate model |

### Options

| Option | Default | Description |
|---|---|---|
| `--threshold / -t` | 0.5 | Score threshold for jet acceptance |
| `--pt-cut` | 200.0 | Minimum pT [GeV] for large-R jets |
| `--dr-match` | 1.4 | ΔR cone for truth particle association |
| `--max-events / -n` | all | Stop after this many events |
| `--max-particles` | 200 | Max particles per jet fed to the model |
| `--output / -o` | — | Write results as JSON to this file |
| `--verbose / -v` | — | Print progress every 100 events |

### Example

```bash
jet-surrogate events.hepmc -m surrogate.onnx -v -o results.json
```

```
Input : events.hepmc
Model : surrogate.onnx
Cuts  : R=1.0 jet pT > 200.0 GeV | dR match < 1.4 | score > 0.5

Events processed       : 10000
Events with >=2 tagged : 3842
Acceptance             : 0.384200
```

## Features

Each truth particle associated to a jet is described by 32 features assembled in this order for the model input tensor `[n_jets, max_particles, 32]`.

### Integer features (indices 0–3)

| Name | Description |
|---|---|
| `pdgId` | PDG particle ID |
| `charge` | Electric charge (integer units) |
| `child0PdgId` | PDG ID of first decay child (0 if none) |
| `child1PdgId` | PDG ID of second decay child (0 if none) |

### Float features (indices 4–31)

| Name | Description |
|---|---|
| `pt` | Transverse momentum [GeV] |
| `mass` | Invariant mass [GeV] |
| `energy` | Energy [GeV] |
| `eta` | Pseudorapidity |
| `phi` | Azimuthal angle [rad] |
| `deta` | η − η_jet |
| `dphi` | Δφ(particle, jet) [rad] |
| `dr` | ΔR(particle, jet) |
| `decayVertexX/Y/Z` | Decay vertex position [mm] (0 if stable) |
| `Lxy` | Transverse decay length from origin [mm] |
| `decayVertexDPhi` | Δφ(decay displacement, particle momentum) |
| `decayVertexDEta` | Δη(decay displacement direction, particle momentum) |
| `prodVertexX/Y/Z` | Production vertex position [mm] |
| `prodLxy` | Transverse production vertex distance from origin [mm] |
| `child0Pt/Eta/Phi/E/M` | Kinematics of first decay child (0 if none) |
| `child1Pt/Eta/Phi/E/M` | Kinematics of second decay child (0 if none) |

## Model interface

The ONNX model is expected to have:

- **Input 0** — `float32[batch, max_particles, 32]` particle feature tensor
- **Input 1** *(optional)* — `float32[batch, max_particles]` validity mask (1 = real particle, 0 = padding)
- **Output 0** — per-jet score; if 2-D `[batch, n_classes]` the last column is used

If your model uses named inputs, update `inference.py:OnnxJetScorer.score()` to map the input names explicitly.
