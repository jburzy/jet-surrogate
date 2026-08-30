"""PyTorch Lightning training stack, modeled on ej-vae: a DataModule, a
LightningModule wrapping ``ParticleTransformer``, per-epoch checkpoints,
Comet logging and YAML configs driven through a ``LightningCLI`` subclass."""
