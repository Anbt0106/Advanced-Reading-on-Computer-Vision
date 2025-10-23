import tensorflow as tf, time

def check_gpu():
    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("CUDA is available.")
        print("Number of GPUs detected:", len(gpus))

        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get("device_name", f"GPU {i}")
            print(f"GPU {i}: {name}")

        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Logical GPUs:", len(logical_gpus))
        print("Current device:", logical_gpus[0].name if logical_gpus else "CPU")

        # Test speed of GPU vs CPU

        with tf.device('/GPU:0'):
            a = tf.random.normal([8192, 8192])
            b = tf.random.normal([8192, 8192])
            start = time.time()
            c = tf.matmul(a, b)
            tf.print("Time on GPU:", time.time() - start, "seconds")

    else:
        print("CUDA is not available. Using CPU.")
        print("Current device: CPU")

if __name__ == "__main__":
    check_gpu()
