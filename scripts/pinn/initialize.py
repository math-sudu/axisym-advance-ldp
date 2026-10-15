"""Generate the common case-B initial equilibrium for neural construction."""
import argparse
import json

from neural_model import ROOT, initialize
import torch
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=["coarse", "production", "design", "interior"],
                        default="production")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        model, source = initialize(args.mesh)
    print(json.dumps({"initial_state": source.relative_to(ROOT).as_posix(),
                      "nodes": model.mesh.n_nodes, "elements": model.mesh.n_elems}))


if __name__ == "__main__":
    main()
