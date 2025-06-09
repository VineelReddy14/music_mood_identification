import tensorflow as tf
import time

# Print TensorFlow version and available GPUs
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

if not gpus:
    raise RuntimeError("No GPU found! Make sure CUDA and cuDNN are properly installed.")

# Set matrix size for a heavy operation
N = 8192  # You can lower this if memory is a concern

print(f"Running heavy matrix multiplication: {N}x{N}")

# Create random tensors
a = tf.random.normal([N, N])
b = tf.random.normal([N, N])

# Perform multiplication on GPU
with tf.device('/GPU:0'):
    start_time = time.time()
    c = tf.matmul(a, b)
    tf.keras.backend.eval(c[0][0])  # Force execution
    elapsed = time.time() - start_time

print(f"Matrix multiplication completed in {elapsed:.2f} seconds")
