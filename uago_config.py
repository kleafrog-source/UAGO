UAGO_CONFIG = {
    "name": "Universal Adaptive Geometric Observer",
    "version": "1.0",
    "language": "en",
    "description": "A fully abstract, technology-agnostic framework for autonomous discovery of deep mathematical structure in any sensory stream.",
    "core_principle": "Rejection of pre-defined object ontologies in favor of pure geometric, topological, and dynamical invariants. The system does not recognize 'things'; it iteratively uncovers the hidden laws that generate the observable patterns.",
    "observation_cycle": [
        {
            "phase": 1,
            "name": "Primary Structure Detection",
            "goal": "Detect coherent spatio-temporal organization against homogeneous background",
            "output": "Region(s) of interest with internal structure"
        },
        {
            "phase": 2,
            "name": "Coarse Invariant Extraction",
            "invariants": [
                "Effective dimensionality",
                "Characteristic scales",
                "Connectivity degree",
                "Presence of repetition",
                "Rough symmetry properties"
            ],
            "output": "Neutral mathematical passport of the phenomenon"
        },
        {
            "phase": 3,
            "name": "Hypothesis Generation",
            "examples": [
                "Possible hierarchical self-similarity",
                "Likely discrete symmetry group",
                "Presence of topological defects",
                "Scale-invariant or multifractal spectrum"
            ],
            "output": "Prioritized list of deeper invariants worth measuring"
        },
        {
            "phase": 4,
            "name": "Adaptive Measurement Request",
            "nature": "Mathematically precise specification of required next-level invariants (angles, curvatures, homology groups, branching statistics, etc.)",
            "property": "Sensor subsystem reconfigures itself to deliver exactly the requested data"
        },
        {
            "phase": 5,
            "name": "Integration & Minimal Model Search",
            "goal": "Find the smallest generative mathematical object capable of reproducing all measured invariants",
            "candidate_models": [
                "Algebraic varieties",
                "Iterated function systems",
                "Discrete dynamical systems",
                "Lie groups or algebras",
                "Category-theoretic descriptions"
            ],
            "output": "Candidate meta-formula"
        },
        {
            "phase": 6,
            "name": "Predictive Validation & Refinement",
            "action": "Generate predictions for yet-unmeasured properties → request verification → refine or replace model",
            "termination": "Saturation: new measurements add no novel information"
        },
        {
            "phase": 7,
            "name": "Scale / Context Transition",
            "options": [
                "Shift to different spatio-temporal scale of the same phenomenon",
                "Transfer attention to adjacent structure using accumulated meta-formulas as prior"
            ]
        }
    ],
    "system_properties": {
        "adaptivity": "Behavior is entirely determined by the mathematical complexity of the observed phenomenon; no fixed protocols or object classes",
        "scale_invariance": "Identical cycle applies from Planck length to cosmological horizons",
        "self_amplification": "Every finalized meta-formula becomes a reusable primitive for future analyses",
        "semantic_delay": "Human-interpretable names appear only at the final communication stage, if required"
    },
    "final_deliverable_per_cycle": {
        "components": [
            "Complete hierarchy of measured invariants",
            "Minimal generative meta-formula",
            "List of non-obvious emergent properties",
            "Stability analysis and possible phase transitions under perturbation"
        ],
        "nature": "Pure mathematical portrait rather than 'object label'"
    },
    "ultimate_purpose": "To create a new kind of knowledge: knowledge not about things, but about the geometric and dynamical essences that things instantiate — directly readable from raw sensory reality.",
    "author": "Conceptual architecture",
    "date": "2025-11-25"
}

DEMO_DATA = {
    "snowflake": {
        "phase1": {
            "roi": {"x": 50, "y": 50, "width": 200, "height": 200},
            "structure_detected": True,
            "complexity_score": 0.87
        },
        "phase2": {
            "dimensionality": 1.26,
            "scales": [2, 4, 8, 16],
            "connectivity": "high",
            "repetition": True,
            "symmetry": "6-fold rotational"
        },
        "phase3": {
            "hypotheses": [
                {"id": "H1", "desc": "Self-similar IFS with 6-fold symmetry", "priority": 0.95},
                {"id": "H2", "desc": "Fractal branching process", "priority": 0.82},
                {"id": "H3", "desc": "Recursive geometric tiling", "priority": 0.71}
            ]
        },
        "phase4": {
            "measurements_requested": ["branch_angles", "scale_ratios", "symmetry_group"],
            "measured_values": {"branch_angles": [60, 120, 240, 300], "scale_ratios": [0.33, 0.33], "symmetry_group": "C6"}
        },
        "phase5": {
            "model": "IFS with 6 affine transformations",
            "formula": "F_i(z) = s*R(60i°)*z + t_i, i=0..5",
            "parameters": {"s": 0.33, "rotation_step": 60}
        },
        "phase6": {
            "predictions": ["Hausdorff dimension ≈ 1.26", "Self-similarity at scales 3^n"],
            "validation_score": 0.94
        },
        "phase7": {
            "next_scale": "microscale crystal structure",
            "context_shift": "adjacent ice formation"
        }
    },
    "spiral": {
        "phase1": {
            "roi": {"x": 30, "y": 30, "width": 240, "height": 240},
            "structure_detected": True,
            "complexity_score": 0.73
        },
        "phase2": {
            "dimensionality": 1.0,
            "scales": [10, 20, 40],
            "connectivity": "continuous",
            "repetition": False,
            "symmetry": "rotational (approximate)"
        },
        "phase3": {
            "hypotheses": [
                {"id": "H1", "desc": "Logarithmic spiral curve", "priority": 0.91},
                {"id": "H2", "desc": "Archimedean spiral", "priority": 0.78},
                {"id": "H3", "desc": "Fibonacci growth pattern", "priority": 0.69}
            ]
        },
        "phase4": {
            "measurements_requested": ["curvature_profile", "angular_velocity", "radius_growth_rate"],
            "measured_values": {"curvature": "exponential", "growth_rate": 1.618, "angular_velocity": "constant"}
        },
        "phase5": {
            "model": "Logarithmic spiral",
            "formula": "r(θ) = a*e^(b*θ), φ = 1.618",
            "parameters": {"a": 1, "b": 0.1762}
        },
        "phase6": {
            "predictions": ["Golden angle spacing", "Scale invariance under rotation"],
            "validation_score": 0.89
        },
        "phase7": {
            "next_scale": "cellular arrangement pattern",
            "context_shift": "neighboring geometric forms"
        }
    },
    "branching": {
        "phase1": {
            "roi": {"x": 100, "y": 50, "width": 180, "height": 200},
            "structure_detected": True,
            "complexity_score": 0.81
        },
        "phase2": {
            "dimensionality": 1.7,
            "scales": [1, 2, 4, 8, 16],
            "connectivity": "tree-like",
            "repetition": True,
            "symmetry": "bilateral (approximate)"
        },
        "phase3": {
            "hypotheses": [
                {"id": "H1", "desc": "L-system with stochastic branching", "priority": 0.88},
                {"id": "H2", "desc": "Diffusion-limited aggregation", "priority": 0.75},
                {"id": "H3", "desc": "Recursive bifurcation process", "priority": 0.82}
            ]
        },
        "phase4": {
            "measurements_requested": ["branching_angles", "length_ratios", "bifurcation_count"],
            "measured_values": {"angles": [25, 35, 45], "length_ratio": 0.7, "generations": 5}
        },
        "phase5": {
            "model": "L-system",
            "formula": "F → F[+F]F[-F]F, angle=30°, ratio=0.7",
            "parameters": {"angle": 30, "ratio": 0.7, "iterations": 5}
        },
        "phase6": {
            "predictions": ["Fractal dimension ≈ 1.7", "Terminal branch count = 2^n"],
            "validation_score": 0.86
        },
        "phase7": {
            "next_scale": "leaf venation details",
            "context_shift": "root system analysis"
        }
    }
}
