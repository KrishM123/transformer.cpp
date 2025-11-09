cmake_minimum_required(VERSION 3.14)

project(TransformerCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")

include_directories(include)

add_executable(neural_network
    src/main.cpp
    src/utils/Tensor.cpp
    src/layers/Linear.cpp
    src/layers/Activations.cpp
    src/utils/Optimizer.cpp
    src/utils/LossFunction.cpp
    src/layers/LayerNorm.cpp
    src/layers/PositionwiseFeedForward.cpp
    src/layers/MultiHeadAttention.cpp
    src/layers/Dropout.cpp
    src/models/DecoderLayer.cpp
    src/models/EncoderLayer.cpp
    src/layers/Embedding.cpp
    src/layers/PositionalEncoding.cpp
    src/utils/Masking.cpp
    src/models/Encoder.cpp
    src/models/Decoder.cpp
    src/models/Transformer.cpp
    src/utils/DataLoader.cpp
    src/utils/ThreadPool.cpp
    src/utils/ConfigParser.cpp
)

if (WIN32)
else()
  # Link filesystem library (C++17)
  target_link_libraries(neural_network PRIVATE stdc++fs)
endif()

add_executable(test_tensor
    tests/test_tensor.cpp
    src/utils/Tensor.cpp
    src/layers/Linear.cpp
    src/utils/ThreadPool.cpp
    src/utils/ConfigParser.cpp
)
