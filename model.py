import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, GlobalAveragePooling2D, Multiply
from tensorflow.keras.models import Model

# Squeeze-and-Excitation Block
def se_block(input_tensor, reduction=16):
    """
    Squeeze-and-Excitation Block.
    Args:
        input_tensor: Tensor from the previous layer.
        reduction: Reduction ratio for bottleneck.
    Returns:
        Tensor after applying SE block.
    """
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

def get_se_cnn_model():
    """
    Returns a CNN model enhanced with Squeeze-and-Excitation blocks.
    """
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = se_block(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = se_block(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Huấn luyện và lưu mô hình
def train_and_save_cnn_model():
    """
    Trains and saves the CNN model for digit classification (0-9).
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create and compile the model
    model = get_se_cnn_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        validation_data=(x_test, y_test),
        batch_size=64,
        verbose=2
    )

    # Save the model
    model.save('se_cnn_mnist_28x28.h5')
    print("Mô hình CNN cải tiến đã được lưu thành công.")

    # Visualize training progress
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

# Run the training process
if __name__ == "__main__":
    train_and_save_cnn_model()
