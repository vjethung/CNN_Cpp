import tensorflow as tf
import numpy as np

# Đọc dữ liệu từ file
def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [float(line.strip()) for line in lines if line.strip()]
    return np.array(data)

# Đường dẫn file
input_file = "input.txt"    # Document 3
weights_file = "weight1.txt" # Document 2
bias_file = "bias1.txt"     # Document 1
output_file = "output.txt"  # Document 4 (để so sánh)

# Đọc dữ liệu
input_data = read_file(input_file)  # 128 giá trị
weights_data = read_file(weights_file)[:640]  # 640 giá trị đầu
bias_data = read_file(bias_file)  # 10 giá trị
expected_output = read_file(output_file)  # 20 giá trị

# Chuẩn bị dữ liệu
input_array = input_data.reshape(-1, 64)  # (2, 64)
weights = weights_data.reshape(10, 64)    # (10, 64) thay vì (64, 10)
weights_transposed = np.transpose(weights)  # (64, 10)

# Tạo lớp Dense
dense_layer = tf.keras.layers.Dense(
    units=10,
    input_shape=(64,),
    kernel_initializer=tf.constant_initializer(weights_transposed),
    bias_initializer=tf.constant_initializer(bias_data),
    use_bias=True
)

# Tạo dataset với batch_size=2
dataset = tf.data.Dataset.from_tensor_slices(input_array).batch(2)

# Tính toán output
print("Output tính toán (TensorFlow):")
for batch in dataset:
    output = dense_layer(batch)
    print(output.numpy())

# So sánh với output mong muốn
expected_output_reshaped = expected_output.reshape(2, 10)
print("\nOutput mong muốn từ file C++:")
print(expected_output_reshaped)