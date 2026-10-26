"""The four models we compare, built the same way for every run."""

from tensorflow.keras import Model, Sequential, layers, optimizers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16

IMG = (224, 224, 3)
NUM_CLASSES = 4
BASES = {"mobilenet": MobileNetV2, "resnet50": ResNet50, "vgg16": VGG16}


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_scratch(lr=1e-3):
    # same CNN as part 1, so old and new results stay comparable
    model = Sequential([
        layers.Input(IMG),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn_scratch")
    return compile_model(model, lr)


def build_transfer(kind, lr=1e-3, dropout=0.3):
    base = BASES[kind](input_shape=IMG, include_top=False, weights="imagenet")
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return compile_model(Model(base.input, out, name=f"{kind}_transfer"), lr)


def unfreeze(model, lr=1e-4, last=30):
    # everything between the input and the 3 head layers is the pretrained base
    body = model.layers[1:-3]
    for layer in body:
        layer.trainable = True
    for layer in body[:-last]:
        layer.trainable = False
    for layer in body:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return compile_model(model, lr)
