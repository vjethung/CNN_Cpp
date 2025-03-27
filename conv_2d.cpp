#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;

void readDataFromFile(const std::string &filename, float* &input) {
    ifstream file(filename);
    int size = 0;
    vector<float> dataVec;
    float value;

    // Đọc từng giá trị từ file và lưu vào vector
    while (file >> value) {
        dataVec.push_back(value);
    }

    file.close();

    // Gán kích thước và cấp phát bộ nhớ cho input
    size = dataVec.size();
    input = new float[size];

    // Sao chép dữ liệu từ vector vào mảng input
    for (int i = 0; i < size; ++i) {
        input[i] = dataVec[i];
    }
}

void saveOutputToFile(const string &filename, const float *output, int batch_size, int output_channels, int height, int width) {
    ofstream file(filename);

    // Duyệt qua tensor và ghi dữ liệu vào file
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < output_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    file << output[b * output_channels * height * width 
                                  + c * height * width 
                                  + h * width 
                                  + w] 
                         << endl;
                }
            }
        }
    }

    file.close();
    cout << "Done " << filename << endl;
}

struct conv2d_shape
{
    int batch_size; // Số lượng mẫu trong một batch
    int channels;   // Số kênh (ví dụ: 3 cho ảnh RGB)
    int height;     // Chiều cao của tensor 
    int width;      // Chiều rộng của tensor 
};

struct kernel_shape 
{
    int size[2]; // size[0]: Chiều cao, size[1]: chiều rộng kernel 
};

struct conv2d_params
{
    float* weight;  //Con trỏ tới mảng trọng số của kernel
    float* bias;
};

// Lop Convolutional
class Conv2d {
private:
    conv2d_shape in_shape, out_shape;
    kernel_shape kernel_sh;
    conv2d_params params;
    int stride[2];
    int padding[2];
    bool use_bias;
public:
    //Định nghĩa hàm khởi tạo conv2d_shape 
    Conv2d(conv2d_shape in_shape, int output_channels, kernel_shape kernel_sh, conv2d_params params, int stride[2], int padding[2], bool use_bias = true) 
    {
        this->in_shape = in_shape;
        this->out_shape.channels = output_channels; // output_chanels = số kernel
        this->kernel_sh = kernel_sh;
        this->params = params;
        this->stride[0] = stride[0]; 
        this->stride[1] = stride[1]; 
        this->padding[0] = padding[0];
        this->padding[1] = padding[1];
        this->use_bias = use_bias;

        //output shape
        int out_height = (in_shape.height + 2 * padding[0] - kernel_sh.size[0]) / stride[0] + 1;
        int out_width = (in_shape.width + 2 * padding[1] - kernel_sh.size[1]) / stride[1] + 1;
        out_shape.height = out_height;     
        out_shape.width = out_width;
        out_shape.batch_size = in_shape.batch_size;
    }

    // Lan truyen xuôi
    void forward(float* input, float* output) {
        // Output shape
        int out_height = (in_shape.height + 2 * padding[0] - kernel_sh.size[0]) / stride[0] + 1;
        int out_width = (in_shape.width + 2 * padding[1] - kernel_sh.size[1]) / stride[1] + 1;
        out_shape.height = out_height;     
        out_shape.width = out_width;
        out_shape.batch_size = in_shape.batch_size; 
    
        // Tính toán output
        for (int b = 0; b < in_shape.batch_size; b++) // Số ảnh = 1 <==> b = 0, sau đó tăng dần
        {
            for (int c = 0; c < out_shape.channels; c++)
            {
                for (int h = 0; h < out_shape.height; h++)
                {
                    for (int w = 0; w < out_shape.width; w++)
                    {
                        float sum = 0;
                        for (int ic = 0; ic < in_shape.channels; ic++)
                        {
                            for (int kh = 0; kh < kernel_sh.size[0]; kh++)
                            {
                                for (int kw = 0; kw < kernel_sh.size[1]; kw++)
                                {
                                    int in_h = h * stride[0] + kh - padding[0];
                                    int in_w = w * stride[1] + kw - padding[1];
                                    if (in_h >= 0 && in_h < in_shape.height && in_w >= 0 && in_w < in_shape.width)
                                    {
                                        // Khai báo chỉ số input
                                        int input_idx = b*in_shape.channels*in_shape.height*in_shape.width + ic*in_shape.height*in_shape.width + in_h*in_shape.width + in_w;
                                        // Khai báo chỉ số weight
                                        int weight_idx = c*in_shape.channels*kernel_sh.size[0]*kernel_sh.size[1] + ic*kernel_sh.size[0]*kernel_sh.size[1] + kh*kernel_sh.size[1] + kw;
                                        // Tính tổng tích
                                        sum +=  input[input_idx] * params.weight[weight_idx];
                                    }
                                }
                            }
                        }
                        // Khai báo chỉ số output và bias
                        int output_idx = b*out_shape.channels*out_shape.height*out_shape.width + c*out_shape.height*out_shape.width + h*out_shape.width + w;
                        // Gán giá trị đầu ra
                        output[output_idx] = sum + (use_bias && params.bias ? params.bias[c] : 0);
                }
            }
        }
    }
    }

void z_score_normalization(double input[], conv2d_shape in_shape, double epsilon) {
    int size = in_shape.height * in_shape.width;
    for (int ic = 0; ic < in_shape.channels; ic++) {

        double mean = 0.0, std_dev = 0.0, sum = 0;

        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.width; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                mean += input[input_idx];
            }
        }

        double mean = sum / size;

        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.height; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                std_dev += (input[input_idx] - mean) * (input[input_idx] - mean);
            }
        }
        std_dev = sqrt(std_dev / size);

        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.height; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                input[input_idx] = (input[input_idx] - mean) / (std_dev + epsilon);
            }
        }

    }
}
void relu( float input[], conv2d_shape in_shape) {
    for (int i = 0; i < in_shape.channels * in_shape.height * in_shape.width; i++) {
        input[i] = fmax(0.0, input[i]);
    }
}
struct pool_shape {
    int height;
    int width;
    int depth;
};
void maxPooling2D(float input[], float output[], conv2d_shape in_shape, int stride[], pool_shape pool) {

    int out_height = (in_shape.height - pool.height) / stride[0] + 1;
    int out_width = (in_shape.width - pool.width) / stride[0] + 1;
    int out_depth = in_shape.channels;

    for (int c = 0; c < out_depth; c++)
    {
        for (int h = 0; h < out_height; h++)
        {
            for (int w = 0; w < out_width; w++)
            {
                int in_start_h = h * stride[0];
                int in_start_w = w * stride[1];
                double max_val = input[c * in_shape.height * in_shape.width + in_start_h * in_shape.width + in_start_w];
                    for (int ph = 0; ph < pool.height; ph++)
                    {
                        for (int pw = 0; pw < pool.width; pw++)
                        {
                            int in_h = in_start_h + ph;
                            int in_w = in_start_w + pw;
                            int input_idx = c * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                            max_val = fmax(max_val, input[input_idx]);

                        }
                    }
                int output_idx = c * out_height * out_width + h * out_width + w;
                output[output_idx] = max_val;

            }
        }
    }
}

    conv2d_shape get_output_shape() const { return out_shape; }
};


int main() {   
    conv2d_shape input_shape = {1, 3, 5, 5};
    kernel_shape kernel_shape = {3, 3};
    conv2d_params params;
    
    // Cac tham so
    int output_channels = 1;
    int padding[] = {1, 1};
    int stride[] = {1, 1};
    bool use_bias = false;

    // Load weight
    string filename1 = "weight.txt";
    readDataFromFile(filename1, params.weight);
     
    // Load bias
    string filename_bias = "bias.txt";
    readDataFromFile(filename_bias, params.bias);

    // Tạo đối tượng Conv2d
    Conv2d conv_layer(input_shape, output_channels, kernel_shape, params, stride, padding, use_bias);

    // Cấp phát input và output
    float* input = new float[input_shape.batch_size * input_shape.channels * input_shape.height * input_shape.width];
    float* output = new float[input_shape.batch_size * output_channels * input_shape.height * input_shape.width];

    // Load input
    string filename2 = "input_C.txt";
    readDataFromFile(filename2, input);
    
    // Chạy forward
    conv_layer.forward(input, output);

    string filename = "output_C.txt";
    
    saveOutputToFile(filename, output, input_shape.batch_size, output_channels, input_shape.height, input_shape.width);
    cout << "(" << conv_layer.get_output_shape().batch_size << "," 
         << conv_layer.get_output_shape().channels << "," 
         << conv_layer.get_output_shape().height << "," 
         << conv_layer.get_output_shape().width << ")" << endl;
    //cout << params.bias[0] << endl;
    delete[] input;
    delete[] output;
    delete[] params.weight; // Giải phóng weight
    delete[] params.bias;  // Giải phóng bias
    return 0;
}
