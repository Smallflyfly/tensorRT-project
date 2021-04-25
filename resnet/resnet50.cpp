//
// Created by fangpf on 2021/4/25.
//

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>


using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "arguments not right" <<endl;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (string(argv[1]) == "-s") {
        IHostMemory*
    }
}