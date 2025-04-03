#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>

typedef struct {
    int batch_size;
    int channels;
    int height;
    int width;
} Feature_map_shape;

typedef struct {
    int size[2];
} Kernel_shape;

typedef struct {
    float* weight;
    float* bias;
} Params;

typedef struct {
    Feature_map_shape in_shape, out_shape;
    Kernel_shape kernel_sh;
    Params params;
    int stride[2];
    int padding[2];
    bool use_bias;
} Layer;

void Layer_init(Layer* conv, Feature_map_shape in_shape, int output_channels, 
                 Kernel_shape kernel_sh, Params params, 
                 int stride[2], int padding[2], bool use_bias) 
{
    conv->in_shape = in_shape;
    conv->out_shape.channels = output_channels;
    conv->kernel_sh = kernel_sh;
    conv->params = params;
    conv->stride[0] = stride[0]; 
    conv->stride[1] = stride[1]; 
    conv->padding[0] = padding[0];
    conv->padding[1] = padding[1];
    conv->use_bias = use_bias;

}

// Forward function
void Conv2d_forward(Layer* conv, float* input, float* output) {
    // Output shape
    int out_height = (conv->in_shape.height + 2 * conv->padding[0] - conv->kernel_sh.size[0]) / conv->stride[0] + 1;
    int out_width = (conv->in_shape.width + 2 * conv->padding[1] - conv->kernel_sh.size[1]) / conv->stride[1] + 1;
    conv->out_shape.height = out_height;     
    conv->out_shape.width = out_width;
    conv->out_shape.batch_size = conv->in_shape.batch_size; 

    // Calculate output
    for (int b = 0; b < conv->in_shape.batch_size; b++)
    {
        for (int c = 0; c < conv->out_shape.channels; c++)
        {
            for (int h = 0; h < conv->out_shape.height; h++)
            {
                for (int w = 0; w < conv->out_shape.width; w++)
                {
                    float sum = 0;
                    for (int ic = 0; ic < conv->in_shape.channels; ic++)
                    {
                        for (int kh = 0; kh < conv->kernel_sh.size[0]; kh++)
                        {
                            for (int kw = 0; kw < conv->kernel_sh.size[1]; kw++)
                            {
                                int in_h = h * conv->stride[0] + kh - conv->padding[0];
                                int in_w = w * conv->stride[1] + kw - conv->padding[1];
                                if (in_h >= 0 && in_h < conv->in_shape.height && in_w >= 0 && in_w < conv->in_shape.width)
                                {
                                    // Input index
                                    int input_idx = b * conv->in_shape.channels * conv->in_shape.height * conv->in_shape.width + 
                                                   ic * conv->in_shape.height * conv->in_shape.width + 
                                                   in_h * conv->in_shape.width + in_w;
                                    // Weight index
                                    int weight_idx = c * conv->in_shape.channels * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] + 
                                                    ic * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] + 
                                                    kh * conv->kernel_sh.size[1] + kw;
                                    // Calculate sum of products
                                    sum += input[input_idx] * conv->params.weight[weight_idx];
                                }
                            }
                        }
                    }
                    // Output and bias index
                    int output_idx = b * conv->out_shape.channels * conv->out_shape.height * conv->out_shape.width + 
                                    c * conv->out_shape.height * conv->out_shape.width + 
                                    h * conv->out_shape.width + w;
                    // Assign output value
                    output[output_idx] = sum + (conv->use_bias && conv->params.bias ? conv->params.bias[c] : 0);
                }   
            }
        }
    }
}

// Backward function
void Conv2d_backward(Layer* conv, float *output_grad, float *input_grad, float learning_rate, float *input) 
{
    int weight_size = Layer_get_size_weight(conv);
    float* kernel_grad = (float*)malloc(weight_size * sizeof(float));

    // Calculate gradient of loss with respect to kernel
    for (int c = 0; c < conv->out_shape.channels; c++)
    {
        for (int ic = 0; ic < conv->in_shape.channels; ic++)
        {
            for (int kh = 0; kh < conv->kernel_sh.size[0]; kh++)
            {
                for (int kw = 0; kw < conv->kernel_sh.size[1]; kw++)
                {
                    float grad_sum = 0;
                    for (int h = 0; h < conv->out_shape.height; h++)
                    {
                        for (int w = 0; w < conv->out_shape.width; w++)
                        {
                            int in_h = h * conv->stride[0] + kh - conv->padding[0];
                            int in_w = w * conv->stride[1] + kw - conv->padding[1];
                            if (in_h >= 0 && in_h < conv->in_shape.height && in_w >= 0 && in_w < conv->in_shape.width)
                            {
                                int input_idx = ic * conv->in_shape.height * conv->in_shape.width + 
                                            in_h * conv->in_shape.width + in_w;
                                int output_idx = c * conv->out_shape.height * conv->out_shape.width + 
                                              h * conv->out_shape.width + w;
                                grad_sum += input[input_idx] * output_grad[output_idx];
                            }
                        }
                    }
                    int idx = c * conv->in_shape.channels * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] + 
                          ic * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] + 
                          kh * conv->kernel_sh.size[1] + kw;
                    kernel_grad[idx] = grad_sum;
                }
            }
        }
    }

    // Update weights
    for (int i = 0; i < weight_size; i++)
    {
        conv->params.weight[i] -= learning_rate * kernel_grad[i];
    }

    // Update bias
    for (int j = 0; j < conv->out_shape.channels * conv->out_shape.height * conv->out_shape.width; j++)
    {
        conv->params.bias[j] -= learning_rate * output_grad[j];
    }

    free(kernel_grad);

}

// Maxpooling function
void Maxpool2d(Layer* maxpool, float* input, float* output, Feature_map_shape in_shape, Kernel_shape kernel_sh, int stride[2], int padding[2]) {
    // Init
    maxpool->in_shape = in_shape;
    maxpool->kernel_sh = kernel_sh;
    maxpool->padding[0] = padding[0];
    maxpool->padding[1] = padding[1];
    maxpool->stride[0] = stride[0]; 
    maxpool->stride[1] = stride[1];

    // Output shape
    int out_height = (maxpool->in_shape.height + 2 * maxpool->padding[0] - maxpool->kernel_sh.size[0]) / maxpool->stride[0] + 1;
    int out_width = (maxpool->in_shape.width + 2 * maxpool->padding[1] - maxpool->kernel_sh.size[1]) / maxpool->stride[1] + 1;
    maxpool->out_shape.batch_size = maxpool->in_shape.batch_size;
    maxpool->out_shape.height = out_height;     
    maxpool->out_shape.width = out_width;
    maxpool->out_shape.channels = in_shape.channels;
    // Maxpool
    for (int b = 0; b < maxpool->in_shape.batch_size; b++)
    {
        for (int ic = 0; ic < maxpool->in_shape.channels; ic++)
        {
            for (int oh = 0; oh < out_height; ++oh)
            {
                for (int ow = 0; ow < out_width; ++ow) 
                {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < kernel_sh.size[0]; ++kh) 
                    {
                        for (int kw = 0; kw < kernel_sh.size[1]; ++kw) 
                        {
                            int ih = oh * stride[0] + kh - padding[0];
                            int iw = ow * stride[1] + kw - padding[1];
                            if (ih >= 0 && ih < maxpool->in_shape.height && iw >= 0 && iw < maxpool->in_shape.width) 
                            {
                                int input_idx = b * maxpool->in_shape.channels * maxpool->in_shape.height * maxpool->in_shape.width + 
                                                ic * maxpool->in_shape.height * maxpool->in_shape.width  + ih * maxpool->in_shape.width + iw;

                                float val = input[input_idx];

                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    int output_idx = b * maxpool->in_shape.channels * out_height * out_width + 
                                    ic * out_height * out_width + 
                                    oh * out_width + ow;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}


// Helper functions
int Layer_get_size_weight(const Layer* layer) { 
    return layer->out_shape.channels * layer->in_shape.channels * layer->kernel_sh.size[0] * layer->kernel_sh.size[1]; 
}

Feature_map_shape Layer_get_output_shape(const Layer* layer) { 
    return layer->out_shape; 
}

// Reverse weight
float* Conv2d_reverse(Layer* conv) {
    int size = conv->out_shape.channels * conv->in_shape.channels * conv->kernel_sh.size[0] * conv->kernel_sh.size[1];
    float* reversed_weight = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        reversed_weight[i] = conv->params.weight[size - 1 - i];
    }
    return reversed_weight;
}

void readDataFromFile(const char *filename, float* *input) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }
    
    int size = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        size++;
    }
    
    rewind(file);
    
    *input = (float*)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &(*input)[i]) != 1) {
            break;
        }
    }
    
    fclose(file);
}

void saveOutputToFile(const char *filename, const float *output, int batch_size, int output_channels, int height, int width) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Iterate through tensor and write data to file
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < output_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    fprintf(file, "%f\n", output[b * output_channels * height * width 
                                                + c * height * width 
                                                + h * width 
                                                + w]);
                }
            }
        }
    }
    
    fclose(file);
    printf("Done %s\n", filename);
}

int main() {
    // Initialize structures
    Feature_map_shape input_shape = {1, 3, 5, 5};
    Kernel_shape kernel_shape = {3, 3};
    Params params = {NULL, NULL};
    
    // Parameters
    int output_channels = 1;
    int padding[] = {1, 1};
    int stride[] = {1, 1};
    int padding_mp[] = {2, 2};
    int stride_mp[] = {3, 3};
    bool use_bias = true;
    
    // Load weight
    char filename1[] = "weight.txt";
    readDataFromFile(filename1, &params.weight);
     
    // Load bias
    char filename_bias[] = "bias.txt";
    readDataFromFile(filename_bias, &params.bias);

    // Create Layer
    Layer conv_layer;
    Layer maxpool;
    Layer_init(&conv_layer, input_shape, output_channels, kernel_shape, params, stride, padding, use_bias);

    // Allocate input and output
    float* input = (float*)malloc(input_shape.batch_size * input_shape.channels * input_shape.height * input_shape.width * sizeof(float));
    float* output = (float*)malloc(input_shape.batch_size * output_channels * input_shape.height * input_shape.width * sizeof(float));
    float* output_MP = (float*)malloc(input_shape.batch_size * output_channels * Layer_get_output_shape(&maxpool).height * Layer_get_output_shape(&maxpool).width * sizeof(float));
    // Load input
    char filename2[] = "input_C.txt";   
    readDataFromFile(filename2, &input);
    
    // Run forward pass
    Conv2d_forward(&conv_layer, input, output);
    
    char filename[] = "output_C.txt";
    saveOutputToFile(filename, output, input_shape.batch_size, output_channels, input_shape.height, input_shape.width);

    Maxpool2d(&maxpool, output, output_MP, Layer_get_output_shape(&conv_layer), kernel_shape, stride_mp, padding_mp);
    // Save output
    char filename3[] = "D:/EDABK/Spikformer/func_Cpp/Maxpooling/output_MP.txt";
    saveOutputToFile(filename3, output_MP, input_shape.batch_size, output_channels, Layer_get_output_shape(&conv_layer).height, Layer_get_output_shape(&conv_layer).width);
    
    // Print output shape
    Feature_map_shape out_shape = Layer_get_output_shape(&conv_layer);
    printf("After conv2d: ");
    printf("(%d,%d,%d,%d)\n", 
           out_shape.batch_size, 
           out_shape.channels, 
           out_shape.height, 
           out_shape.width);
   
    Feature_map_shape out_shape_mp = Layer_get_output_shape(&maxpool);
    printf("After maxpool2d: ");
    printf("(%d,%d,%d,%d)\n", 
            out_shape_mp.batch_size, 
            out_shape_mp.channels, 
            out_shape_mp.height, 
            out_shape_mp.width);
    // Print weights
    //int weight_size = Layer_get_size_weight(&conv_layer);
    
    // Get and print reversed weights
    /*float* reversed = Layer_reverse(&conv_layer);
    for (int i = 0; i < weight_size; i++) {
        printf("%.1f ", reversed[i]);
    }
    */
    // Clean up
    free(input);
    free(output);
    free(params.weight);
    free(params.bias);
    //free(reversed);
    
    return 0;
}