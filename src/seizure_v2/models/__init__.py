from seizure_v2.models.networks import V1CNN, ResidualCNN


def make_model(name):
    if name not in {"baseline", "residual"}:
        raise ValueError(f"Unknown model: {name}")
    return V1CNN() if name == "baseline" else ResidualCNN()
