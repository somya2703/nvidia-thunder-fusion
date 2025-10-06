#include <torch/extension.h>
#include <cuda_runtime.h>

// Minimal Conv + Bias + ReLU kernel example
// NOTE: This is a simple illustrative kernel for testing the build
__global__ void conv_bn_relu_kernel(const float* x, const float* bias, float* y, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int c = idx % C;
        float val = x[idx] + bias[c];  // add bias per channel
        y[idx] = val > 0 ? val : 0;    // ReLU
    }
}

void conv_bn_relu_launcher(torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    int N = x.size(0);
    int C = bias.size(0);
    int total = x.numel();
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv_bn_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), N, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_bn_relu_launcher", &conv_bn_relu_launcher, "Conv+Bias+ReLU fused kernel");
}
