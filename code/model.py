import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_mlp_model(input_dim=1024, output_dim=56, dropout1=0.3, dropout2=0.3, dropout3=0.2):
    """
    Builds and returns a compiled MLP model for multi-label mood classification.

    Parameters:
    - input_dim: Dimension of input features (default 1024 for YAMNet embeddings)
    - output_dim: Number of mood tags (default 56)
    - dropout1/2/3: Dropout rates for each hidden layer

    Returns:
    - Compiled tf.keras model
    """
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout1),

        Dense(256, activation='relu'),
        Dropout(dropout2),

        Dense(128, activation='relu'),
        Dropout(dropout3),

        Dense(output_dim, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc')
        ]
    )

    return model