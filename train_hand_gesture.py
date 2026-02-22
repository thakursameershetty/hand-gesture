import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_model(num_classes, img_size, fine_tune_at):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.1)(x)   # Slightly increased from 0.05
    x = tf.keras.layers.RandomZoom(0.15)(x)      # Slightly increased from 0.1

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return model


def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in dataset:
        batch_counts = np.bincount(labels.numpy(), minlength=num_classes)
        counts += batch_counts

    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return {i: float(weights[i]) for i in range(num_classes)}


def main():
    parser = argparse.ArgumentParser(description="Train hand gesture model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Force CPU-only execution as requested
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "ml" / "data" / "handgestures_dataset"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "ml" / "hand_gesture" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected train/ and test/ directories in handgestures_dataset")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset="training",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset="validation",
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
    with (output_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    model = build_model(len(class_names), args.img_size, args.fine_tune_at)
    class_weights = compute_class_weights(train_ds, len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            monitor="val_accuracy",
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.save(output_dir / "model.keras")

    results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
