import tensorflow as tf

def build_core_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Layer 1: Convolutional layer
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 2: Convolutional layer
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 3: Convolutional layer
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 4: Convolutional layer
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 5: Convolutional layer
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 6: Convolutional layer
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten layer
    x = tf.keras.layers.Flatten()(x)
    
    # Fully connected layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
