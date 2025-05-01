#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <memory>
#include <stdexcept>

#include "utils/Tensor.h"

extern std::mt19937 global_rng;

class DataLoader {
public:
    // Constructor
    DataLoader(const std::string& filename, int sequence_length, int batch_size);

    // Load and preprocess the data
    void load_data();

    // Get vocabulary size
    int get_vocab_size() const;

    // Get a random batch of data
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_batch();

    // Get character from ID
    char get_char_from_id(int id) const;

    // Get ID from character
    int get_id_from_char(char character) const;

private:
    std::string filename_;
    int sequence_length_;
    int batch_size_;

    std::vector<char> chars_;
    std::unordered_map<char, int> char_to_id_;
    std::unordered_map<int, char> id_to_char_;
    std::vector<int> data_;

    size_t num_batches_;
};

#endif // DATALOADER_H
