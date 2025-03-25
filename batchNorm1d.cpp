#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

void readDataFromFile(const std::string &filename, float *&input, int &size) {
    ifstream file(filename);
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
    cout << "Done" << filename << endl;
}


struct batchNorm1d_shape
{
    int batch_size;
    int channels;
    int patchs;
};

struct batchNorm1d_weight
{
    float *weight;
    float *mean;
    float *var;
};

class BatchNorm1d
{
public:
    BatchNorm1d(batchNorm1d_shape shape, batchNorm1d_weight weight, float eps);
    void update_weight(batchNorm1d_weight weight);
    void forward(float *input, float *output);

private:
    batchNorm1d_shape shape;
    batchNorm1d_weight weight;
    float eps;
    float *input;
    float *output;
};

BatchNorm1d::BatchNorm1d(batchNorm1d_shape shape, batchNorm1d_weight weight, float eps = 1e-5)
    {
        this->shape = shape;
        this->weight = weight;
        this->eps = eps;
    }

    void BatchNorm1d::update_weight(batchNorm1d_weight weight)
    {
        this->weight = weight;
    }

    void BatchNorm1d::forward(float *input, float *output)
    {
        this->input = input;
        this->output = output;

        for (int b = 0; b < shape.batch_size; b++)
        {
            for (int c = 0; c < shape.channels; c++)
            {
                for (int p = 0; p < shape.patchs; p++)
                {
                    output[b * shape.channels * shape.patchs + c * shape.patchs + p] = (input[b * shape.channels * shape.patchs + c * shape.patchs + p] - weight.mean[c]) / sqrt(weight.var[c] + eps) * weight.weight[c];
                }
            }
        }
    }

int main () {
    float *input = NULL;
    int size = 0;
    readDataFromFile("input_B.txt", input, size);
    cout << "Dữ liệu đọc được từ file:" << endl;
    for (int i = 0; i < size; i++) {
        cout << input[i] << endl;
    }

    // Giải phóng bộ nhớ sau khi sử dụng
    delete[] input;
} 