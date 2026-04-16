"""Custom Prometheus metrics for the fraud detection serving layer."""

from prometheus_client import Counter, Histogram

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds, labelled by model",
    ["model_name"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

INFERENCE_TOTAL = Counter(
    "inference_total",
    "Total completed inference requests",
    ["model_name", "prediction"],  # prediction: "fraud" | "legit"
)

INFERENCE_ERRORS = Counter(
    "inference_errors_total",
    "Total inference errors",
    ["model_name"],
)

AB_ASSIGNMENTS = Counter(
    "ab_test_assignments_total",
    "A/B test model assignments",
    ["model_variant"],  # "champion" | "challenger"
)
