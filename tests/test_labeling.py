import numpy as np

from auto_labeling.labeling import apply_cluster_mapping, map_clusters_to_classes, optimal_label_accuracy


def test_mapping_is_invariant_to_cluster_identifier_permutation():
    clusters = np.array([0, 0, 1, 1, 1, 0])
    labels = np.array([0, 0, 1, 1, 1, 0])
    mapping = map_clusters_to_classes(clusters, labels)
    renamed_clusters = 1 - clusters
    renamed_mapping = map_clusters_to_classes(renamed_clusters, labels)
    assert np.array_equal(apply_cluster_mapping(clusters, mapping), apply_cluster_mapping(renamed_clusters, renamed_mapping))


def test_optimal_accuracy_handles_swapped_clusters():
    clusters = np.array([1, 1, 0, 0])
    labels = np.array([0, 0, 1, 1])
    assert optimal_label_accuracy(clusters, labels) == 1.0
