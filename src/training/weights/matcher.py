"""
Weights matchers: decide which trained weights an evaluation runs together.
`OriginPartitionMatcher` groups them by origin partition, so evaluation
mirrors the origin's partitions whatever its fold structure. It never parses
partition names.
"""
from typing import Dict, List

from src.training.weights.resolver import TrainedWeights


class OriginPartitionMatcher:
    def match(self, trained_weights: List[TrainedWeights]) -> Dict[str, List[TrainedWeights]]:
        """Origin partition -> its weights (one per origin seed), in the
        order the resolver yields them."""
        matched: Dict[str, List[TrainedWeights]] = {}
        for w in trained_weights:
            matched.setdefault(w.partition, []).append(w)
        return matched


MATCHER_REGISTRY = {
    "OriginPartitionMatcher": OriginPartitionMatcher,
}
