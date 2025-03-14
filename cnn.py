import numpy as np

def conv2d_numpy(input_tensor: np.ndarray,
              kernel: np.ndarray) -> np.ndarray:

    h, w, in_channels = input_tensor.shape
    out_channels = kernel.shape[0]

    # Hint: check np.pad() or pad the input by yourself; use three nested for loops to loop over height, width, out_channels
    # Hint: identify how many pixels should be padded, identify the input patch and kernel for given locations (loop over h, w) and out channel (loop over out_channels)
    # ----------- <Your code> ---------------

    kernel_h = kernel.shape[1]
    kernel_w = kernel.shape[2]
    input_tensor = np.pad(input_tensor, pad_width=1, mode='constant', constant_values=0)
    output = np.empty((h,w,out_channels))
    for i in range(0, out_channels):
      temp_output = np.zeros((h,w,1))
      output_j = 0
      for j in range(0, h):
        output_k = 0
        for k in range(0, w):
          # print(input_tensor[k:k + kernel_w, j:j + kernel_h, :].shape)
          # print(kernel[i].shape)
          result = np.sum(np.multiply(input_tensor[k:k + kernel_w, j:j + kernel_h, :], kernel[i]))
          output[output_k][output_j][i] = result
          output_k+=1
        output_j+=1

    # --------- <End your code> -------------

    return output

input_tensor = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]).reshape(4, 4, 1)

# Kernel: 2 output channels, 3x3 size, 1 input channel
kernel = np.zeros((2, 3, 3, 1))
# First kernel: all ones (blur/average)
kernel[0] = np.ones((3, 3, 1))
# Second kernel: horizontal edge detection [1,0,-1] pattern
kernel[1] = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]).reshape(3, 3, 1)

# Expected output shape: (4, 4, 2)
expected_output = np.array([
    [[14, -8], [24, -4], [30, -4], [22, 10]],
    [[33, -18], [54, -6], [63, -6], [45, 21]],
    [[57, -30], [90, -6], [99, -6], [69, 33]],
    [[46, -24], [72, -4], [78, -4], [54, 26]]
])

output = conv2d_numpy(input_tensor, kernel)
assert (output == expected_output).all()