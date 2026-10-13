"""Differentiable near-zone elastoplastic FE core (batch 1).

Plane-strain perturbation problem around a deep circular tunnel on a
structured Q4 annulus, tension-positive frame, PyTorch float64 CPU,
end-to-end differentiable through the multi-increment elastoplastic
solve.  Mirrors the discrete choices of the frozen linear-elastic
prototype ``experiments/coupling_schwarz_proto`` (read-only reuse via
``proto_bridge``).
"""
