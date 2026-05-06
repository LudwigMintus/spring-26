import numpy as np


def calculate_gini_impurity(labels):
    if len(labels) == 0:
        return 0

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    total = len(labels)
    impurity = 1 - sum((count / total) ** 2 for count in counts.values())
    return impurity


def is_numeric_column(values):
    try:
        values.astype(float)
        return True
    except (ValueError, TypeError):
        return False


def calculate_gini_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0

    probabilities = counts / total
    return 1 - np.sum(probabilities ** 2)


def calculate_numeric_information_gain(feature_values, y):
    numeric_values = feature_values.astype(float)
    order = np.argsort(numeric_values)
    sorted_values = numeric_values[order]
    sorted_y = y[order]

    if len(np.unique(sorted_values)) <= 1:
        return 0, None

    _, encoded_y = np.unique(sorted_y, return_inverse=True)
    sorted_encoded_y = encoded_y
    total_counts = np.bincount(sorted_encoded_y)
    left_counts = np.zeros_like(total_counts)
    parent_gini = calculate_gini_from_counts(total_counts)
    best_gain = 0
    best_threshold = None

    for i in range(len(sorted_values) - 1):
        left_counts[sorted_encoded_y[i]] += 1

        if sorted_values[i] == sorted_values[i + 1]:
            continue

        right_counts = total_counts - left_counts
        left_weight = left_counts.sum() / len(y)
        right_weight = right_counts.sum() / len(y)
        weighted_child_gini = (
            left_weight * calculate_gini_from_counts(left_counts)
            + right_weight * calculate_gini_from_counts(right_counts)
        )
        gain = parent_gini - weighted_child_gini

        if gain > best_gain:
            best_gain = gain
            best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

    return best_gain, best_threshold


def calculate_information_gain(X, y, feature_index):
    feature_values = X[:, feature_index]

    if is_numeric_column(feature_values):
        gain, threshold = calculate_numeric_information_gain(feature_values, y)
        return gain, threshold, True

    parent_gini = calculate_gini_impurity(y)
    values, counts = np.unique(feature_values, return_counts=True)
    weighted_child_gini = 0

    for value, count in zip(values, counts):
        subset_y = y[feature_values == value]
        weighted_child_gini += (count / len(y)) * calculate_gini_impurity(subset_y)

    return parent_gini - weighted_child_gini, None, False
