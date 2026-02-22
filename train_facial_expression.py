import argparse
import json
import os
from pathlib import Path

# CPU only (remove if GPU available)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Model Architecture (SqueezeNet - Ultra Fast & Lightweight)
# --------------------------------------------------
def build_model(num_classes, img_size, fine_tune_at):
    preprocess = tf.keras.applications.mobilenet.preprocess_input

    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))

    # Data augmentation
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.15)(x)
    x = tf.keras.layers.RandomContrast(0.15)(x)

    x = preprocess(x)

    # MobileNet (lightweight) - trains 10x faster
    base_model = tf.keras.applications.MobileNet(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.5  # Reduce model width for speed
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Lightweight classifier
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(0.001)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return model


# --------------------------------------------------
# Class Weights (handles imbalance)
# --------------------------------------------------
def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes)

    for _, labels in dataset.unbatch():
        counts[int(labels.numpy())] += 1

    total = np.sum(counts)
    weights = total / (num_classes * counts)
    class_weights = {i: float(weights[i]) for i in range(num_classes)}

    print("Class distribution:", counts)
    print("Class weights:", class_weights)

    return class_weights


# --------------------------------------------------
# Main Training
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Facial Emotion Model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--fine-tune-at", type=int, default=80)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "ml" / "data" / "emotions_dataset"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "ml" / "facial_expression" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected train/ and test folders")

    # --------------------------------------------------
    # Dataset Loading
    # --------------------------------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Save labels
    with (output_dir / "labels.json").open("w") as f:
        json.dump(class_names, f, indent=2)

    # Performance optimization (safe for CPU)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # --------------------------------------------------
    # Build Model
    # --------------------------------------------------
    model = build_model(len(class_names), args.img_size, args.fine_tune_at)

    class_weights = compute_class_weights(train_ds, len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=4,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # --------------------------------------------------
    # Stage 1 Training
    # --------------------------------------------------
    print("\nStage 1: Feature extraction...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # --------------------------------------------------
    # Stage 2 Fine Tuning
    # --------------------------------------------------
    print("\nStage 2: Fine tuning...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(output_dir / "model.keras")

    # --------------------------------------------------
    # Evaluation (Accuracy %)
    # --------------------------------------------------
    print("\nEvaluating on Test Dataset...")
    results = model.evaluate(test_ds)
    metrics = dict(zip(model.metrics_names, results))

    # Handle different possible metric names
    accuracy_key = "accuracy" if "accuracy" in metrics else list(metrics.keys())[1]
    accuracy_value = metrics[accuracy_key]
    accuracy_percentage = accuracy_value * 100
    
    # Store with standard naming
    metrics["accuracy"] = accuracy_value
    metrics["accuracy_percentage"] = round(accuracy_percentage, 2)

    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest Results:")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Accuracy (%): {accuracy_percentage:.2f}%")


if __name__ == "__main__":
    main()
