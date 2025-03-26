#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;
//struct data_shape
//{
//    int num_data;   //height = số dữ liệu 
//    int num_para;   //weight = số node_prelayer(input) or số node(output)
//};
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

void saveOutputToFile(const string &filename, const float *output, int batch_size, int num_nodes) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing!" << endl;
        return;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_nodes; i++) {
            file << output[b * num_nodes + i] << endl;
        }
    }

    file.close();
    cout << "Done saving to " << filename << endl;
}

struct layer_params
{
    float* weight;
    float* bias;
};

//lớp fully_connected
class fullyConnected
{
private:
    //data_shape in_shape, out_shape;
    int num_NodePreLayer, num_NodeThisLayer;
    int batch_size;
    layer_params params;
    bool use_bias;
public:
    //định nghĩa hàm khởi tạo
    fullyConnected(int num_NodePreLayer, int num_NodeThisLayer, int batch_size, layer_params params, bool use_bias = true)
    {
        this->num_NodePreLayer = num_NodePreLayer;
        this->num_NodeThisLayer = num_NodeThisLayer;
        this->batch_size = batch_size;
        this->params = params;
        this->use_bias = use_bias;
    }

    void forward(float* input, float* output) 
    {
        // Tính toán đầu ra cho từng mẫu trong batch
        for (int b = 0; b < batch_size; b++)
        {
            // Con trỏ để trỏ đến input và output của từng batch
            float* current_input = input + (b * num_NodePreLayer);
            float* current_output = output + (b * num_NodeThisLayer);
            for (int i = 0; i < num_NodeThisLayer; i++) // Duyệt qua từng node của lớp hiện tại
            {
                float sum = 0;
                for (int j = 0; j < num_NodePreLayer; j++) // Duyệt qua từng node của lớp trước
                {
                    // Chỉ số của weight
                    int weight_idx = i * num_NodePreLayer + j;
                    sum += current_input[j] * params.weight[weight_idx];
                }
                current_output[i] = sum + (use_bias ? params.bias[i] : 0);
            }
        }
    }
    ~fullyConnected() {}
};

int main() {
    // 1. Thiết lập tham số
    int num_NodePreLayer = 64;  // Đầu vào từ Dense(64)
    int num_NodeThisLayer = 10; // Đầu ra của lớp cuối (10 lớp CIFAR-10)
    int batch_size = 2;         // 2 mẫu trong file input.txt

    // 2. Đọc dữ liệu từ file
    float* weights = nullptr;
    float* biases = nullptr;
    float* input = nullptr;

    readDataFromFile("weight1.txt", weights); // 640 giá trị
    readDataFromFile("bias1.txt", biases);     // 10 giá trị
    readDataFromFile("input.txt", input);     // 128 giá trị

    // Kiểm tra kích thước dữ liệu
    if (!weights || !biases || !input) {
        cout << "Error: Failed to read one or more files!" << endl;
        return 1;
    }

    layer_params params = {weights, biases};

    // 3. Cấp phát bộ nhớ cho output
    float* output = new float[batch_size * num_NodeThisLayer]; // 2 x 10 = 20 giá trị

    // 4. Tạo đối tượng fullyConnected
    fullyConnected fc(num_NodePreLayer, num_NodeThisLayer, batch_size, params, true);

    // 5. Chạy forward pass
    fc.forward(input, output);

    // 6. In kết quả ra màn hình (tùy chọn)
    cout << "Output:" << endl;
    for (int b = 0; b < batch_size; b++) {
        cout << "Batch " << b << ": ";
        for (int i = 0; i < num_NodeThisLayer; i++) {
            cout << output[b * num_NodeThisLayer + i] << " ";
        }
        cout << endl;
    }

    // 7. Lưu kết quả vào file
    saveOutputToFile("output.txt", output, batch_size, num_NodeThisLayer);

    // 8. Giải phóng bộ nhớ
    delete[] weights;
    delete[] biases;
    delete[] input;
    delete[] output;

    return 0;
}
