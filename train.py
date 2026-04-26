import numpy as np
from network import ConvNet, cross_entropy_loss


def load_mnist():
    """Load MNIST via scikit-learn. Downloads on first run (~12MB)."""
    print("Loading MNIST dataset...")
    from sklearn.datasets import fetch_openml
    dataset = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    x = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(int)

    x_train, y_train = x[:60000], y[:60000]
    x_test, y_test = x[60000:], y[60000:]

    # reshape flat vectors to image format (N, channels, H, W)
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    print(f"  train: {x_train.shape}, test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out


def get_batches(x, y, batch_size):
    n = x.shape[0]
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


def accuracy(model, x, y, batch_size=64):
    correct = 0
    for xb, yb in get_batches(x, y, batch_size):
        preds = model.predict(xb)
        correct += np.sum(preds == yb)
    return correct / len(y)


def train(num_train=2000, num_test=500, epochs=5, lr=0.001, batch_size=32):
    x_train, y_train, x_test, y_test = load_mnist()

    # use a subset - full MNIST is slow without GPU
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_test, y_test = x_test[:num_test], y_test[:num_test]

    model = ConvNet()

    print(f"\nTraining on {num_train} samples for {epochs} epochs")
    print(f"lr={lr}, batch_size={batch_size}\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for xb, yb in get_batches(x_train, y_train, batch_size):
            yb_hot = one_hot(yb)

            probs = model.forward(xb)
            loss = cross_entropy_loss(probs, yb_hot)

            model.backward(yb_hot)
            model.update(lr)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_acc = accuracy(model, x_train, y_train)
        test_acc = accuracy(model, x_test, y_test)

        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    train()
