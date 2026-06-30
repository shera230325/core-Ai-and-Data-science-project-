"""
train.py
Trains the soil nutrient CNN and saves the model + accuracy plot.

Usage:
    python train.py
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import get_datasets
from model import build_model

EPOCHS = 15


def main():
    train_ds, val_ds = get_datasets("dataset")

    model = build_model(num_classes=4)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training started...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # Save the trained model
    model.save("models/trained_model.h5")
    print("Model saved to models/trained_model.h5")

    # Plot and save accuracy curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Soil Nutrient CNN — Training Accuracy")
    plt.legend()
    plt.savefig("results/accuracy.png")
    print("Accuracy plot saved to results/accuracy.png")

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final validation accuracy: {final_val_acc:.2%}")


if __name__ == "__main__":
    main()
