import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """Load dataset from a whitespace-delimited text file.

    The first column is the class label; remaining columns are features.
    Returns a tuple (labels, features) as numpy arrays.
    """
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return labels, features


def normalize(features):
    """Z-score normalize each feature column."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1  # avoid division by zero for constant features
    return (features - mean) / std


def leave_one_out_accuracy(labels, features, feature_set):
    """Evaluate a feature subset using leave-one-out cross-validation
    with a 1-nearest-neighbour classifier.

    Parameters
    ----------
    labels : 1-D array of class labels (length n)
    features : 2-D array of all features (n x d)
    feature_set : list/set of 0-based column indices to use

    Returns
    -------
    Accuracy as a float in [0, 1].
    """
    if not feature_set:
        # No features selected – predict majority class for every instance
        majority = np.bincount(labels).argmax()
        return np.mean(labels == majority)

    cols = list(feature_set)
    X = features[:, cols]
    n = len(labels)
    correct = 0

    for i in range(n):
        # Compute distances from instance i to all others
        diff = X - X[i]
        distances = np.sqrt((diff ** 2).sum(axis=1))
        distances[i] = np.inf  # exclude the instance itself
        nn = np.argmin(distances)
        if labels[nn] == labels[i]:
            correct += 1

    return correct / n


def forward_selection(labels, features):
    """Greedy forward feature selection.

    Returns
    -------
    best_set   : set of 0-based feature indices with highest accuracy seen
    best_acc   : the accuracy achieved by best_set
    history    : list of (accuracy, feature_set_snapshot) at each level
    """
    n_features = features.shape[1]
    current_set = set()
    best_set = set()
    best_acc = 0.0
    history = []

    print("\n--- Forward Selection ---")
    print(f"Starting with no features (default accuracy = "
          f"{leave_one_out_accuracy(labels, features, current_set):.4f})\n")

    for level in range(n_features):
        best_feature = None
        level_best_acc = -1.0

        for f in range(n_features):
            if f in current_set:
                continue
            candidate = current_set | {f}
            acc = leave_one_out_accuracy(labels, features, candidate)
            # 1-based feature numbers for display
            print(f"  Using feature(s) {sorted(f + 1 for f in candidate)}"
                  f" -> accuracy = {acc:.4f}")
            if acc > level_best_acc:
                level_best_acc = acc
                best_feature = f

        current_set.add(best_feature)
        history.append((level_best_acc, frozenset(current_set)))

        print(f"\nFeature set {sorted(f + 1 for f in current_set)} is best "
              f"at level {level + 1} (accuracy = {level_best_acc:.4f})\n")

        if level_best_acc > best_acc:
            best_acc = level_best_acc
            best_set = frozenset(current_set)

    print(f"Best feature set: {sorted(f + 1 for f in best_set)} "
          f"(accuracy = {best_acc:.4f})")
    return best_set, best_acc, history


def backward_elimination(labels, features):
    """Greedy backward feature elimination.

    Returns
    -------
    best_set   : set of 0-based feature indices with highest accuracy seen
    best_acc   : the accuracy achieved by best_set
    history    : list of (accuracy, feature_set_snapshot) at each level
    """
    n_features = features.shape[1]
    current_set = set(range(n_features))
    best_set = frozenset(current_set)
    best_acc = leave_one_out_accuracy(labels, features, current_set)
    history = [(best_acc, frozenset(current_set))]

    print("\n--- Backward Elimination ---")
    print(f"Starting with all features (accuracy = {best_acc:.4f})\n")

    for level in range(n_features - 1):
        worst_feature = None
        level_best_acc = -1.0

        for f in list(current_set):
            candidate = current_set - {f}
            acc = leave_one_out_accuracy(labels, features, candidate)
            print(f"  Removing feature {f + 1}, using "
                  f"{sorted(g + 1 for g in candidate)}"
                  f" -> accuracy = {acc:.4f}")
            if acc > level_best_acc:
                level_best_acc = acc
                worst_feature = f

        current_set.discard(worst_feature)
        history.append((level_best_acc, frozenset(current_set)))

        print(f"\nFeature set {sorted(f + 1 for f in current_set)} is best "
              f"at level {level + 1} (accuracy = {level_best_acc:.4f})\n")

        if level_best_acc > best_acc:
            best_acc = level_best_acc
            best_set = frozenset(current_set)

    print(f"Best feature set: {sorted(f + 1 for f in best_set)} "
          f"(accuracy = {best_acc:.4f})")
    return best_set, best_acc, history


def plot_accuracy(forward_history=None, backward_history=None):
    """Plot accuracy vs. number of features for one or both search strategies.

    Either *forward_history* or *backward_history* (or both) must be provided.
    """
    plt.figure(figsize=(8, 5))

    if forward_history:
        fwd_accs = [acc for acc, _ in forward_history]
        fwd_sizes = list(range(1, len(fwd_accs) + 1))
        plt.plot(fwd_sizes, fwd_accs, marker='o', label='Forward Selection')

    if backward_history:
        bwd_accs = [acc for acc, _ in backward_history]
        # backward_history[0] = all features; each step removes one feature
        n_features = len(backward_history[0][1])
        bwd_sizes = list(range(n_features, n_features - len(bwd_accs), -1))
        plt.plot(bwd_sizes, bwd_accs, marker='s', linestyle='--',
                 label='Backward Elimination')

    plt.xlabel('Number of Features')
    plt.ylabel('Leave-One-Out Accuracy')
    plt.title('Feature Selection – Accuracy vs. Number of Features')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=150)
    plt.show()
    print("Plot saved to accuracy_plot.png")


def main():
    print("CS170 Project 2 – Feature Selection with Nearest Neighbor\n")
    print("=" * 60)

    filename = input("Enter the name of the dataset file: ").strip()
    labels, features = load_data(filename)
    features = normalize(features)

    n, d = features.shape
    print(f"\nDataset loaded: {n} instances, {d} features.")
    print(f"Class distribution: { {c: int((labels == c).sum()) for c in np.unique(labels)} }")

    algorithm = input(
        "\nSelect algorithm:\n"
        "  1) Forward Selection\n"
        "  2) Backward Elimination\n"
        "  3) Both\n"
        "Choice: "
    ).strip()

    forward_history = None
    backward_history = None

    if algorithm in ("1", "3"):
        _, _, forward_history = forward_selection(labels, features)

    if algorithm in ("2", "3"):
        _, _, backward_history = backward_elimination(labels, features)

    if forward_history or backward_history:
        plot_accuracy(forward_history, backward_history)


if __name__ == "__main__":
    main()
