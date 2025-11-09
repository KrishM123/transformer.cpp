#include "models/Decoder.h"
#include "models/Encoder.h"
#include "models/Transformer.h"
#include "utils/ConfigParser.h"
#include "utils/DataLoader.h"
#include "utils/Helpers.h"
#include "utils/LossFunction.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

std::mt19937 global_rng(
    std::chrono::high_resolution_clock::now().time_since_epoch().count());

// Function to convert string to tensor of IDs
std::shared_ptr<Tensor> string_to_tensor(const std::string &text,
                                         const DataLoader &loader,
                                         int seq_len) {
  const auto &char_to_id = loader.get_char_to_id_map();
  std::vector<float> ids;
  ids.reserve(seq_len);
  for (char c : text) {
    auto it = char_to_id.find(c);
    if (it != char_to_id.end()) {
      ids.push_back(static_cast<float>(it->second));
    } else {
      ids.push_back(0.0f);
    }
  }
  // Pad
  while (ids.size() < seq_len) {
    ids.push_back(0.0f);
  }
  if (ids.size() > seq_len) {
    ids.resize(seq_len);
  }

  // Create tensor with shape (1, seq_len) for batch size 1
  return Tensor::create({1, seq_len},
                        std::make_shared<std::vector<float>>(ids));
}

int main() {
  ConfigParser::getInstance("../config.ini");
  bool inference_mode =
      ConfigParser::getInstance().getValue<bool>("inference_mode");
  bool load_existing_weights =
      ConfigParser::getInstance().getValue<bool>("load_existing_weights");
  std::string weights_filename =
      ConfigParser::getInstance().getValue<std::string>("weights_filename");
  std::string data_filename =
      ConfigParser::getInstance().getValue<std::string>("data_filename");

  // Model Architecture
  int embed_dim = ConfigParser::getInstance().getValue<int>("embed_dim");
  int max_sequence_length =
      ConfigParser::getInstance().getValue<int>("max_sequence_length");
  int num_layers = ConfigParser::getInstance().getValue<int>("num_layers");
  int num_heads = ConfigParser::getInstance().getValue<int>("num_heads");
  int ff_hidden_dim =
      ConfigParser::getInstance().getValue<int>("ff_hidden_dim");
  float dropout_rate =
      ConfigParser::getInstance().getValue<float>("dropout_rate");
  float pad_token_id =
      ConfigParser::getInstance().getValue<float>("pad_token_id");

  // Training Parameters
  float learning_rate =
      ConfigParser::getInstance().getValue<float>("learning_rate");
  int num_epochs = ConfigParser::getInstance().getValue<int>("num_epochs");
  int batch_size = ConfigParser::getInstance().getValue<int>("batch_size");
  int input_seq_length =
      ConfigParser::getInstance().getValue<int>("input_seq_length");
  int decoder_seq_length =
      ConfigParser::getInstance().getValue<int>("decoder_seq_length");

  // Inference Parameters
  int max_generate_length =
      ConfigParser::getInstance().getValue<int>("max_generate_length");
  std::string initial_prompt =
      ConfigParser::getInstance().getValue<std::string>("initial_prompt");

  // DataLoader needs sequence length and batch size from config
  DataLoader data_loader(data_filename, input_seq_length, batch_size);
  data_loader.load_data();

  int input_vocab_size = data_loader.get_vocab_size();
  int target_vocab_size = data_loader.get_vocab_size();

  Transformer model(input_vocab_size, target_vocab_size, embed_dim,
                    max_sequence_length, num_layers, num_heads, ff_hidden_dim,
                    dropout_rate, pad_token_id);

  // Print model parameters count
  size_t total_params = 0;
  const auto &parameters = Tensor::get_optimizable_tensors();
  for (const auto &param : parameters) {
    if (param) {
      total_params += param->num_elements();
    }
  }
  std::cout << "Total optimizable parameters: " << total_params << std::endl;

  // Load Weights
  bool weights_loaded = false;
  if ((load_existing_weights || inference_mode) &&
      std::filesystem::exists(weights_filename)) {
    try {
      std::cout << "Attempting to load weights from " << weights_filename
                << "..." << std::endl;
      model.load_weights(weights_filename);
      std::cout << "Weights loaded successfully." << std::endl;
      weights_loaded = true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to load weights: " << e.what() << std::endl;
      if (inference_mode) {
        std::cerr << "Cannot run inference without weights. Exiting."
                  << std::endl;
        return 1;
      }
      std::cerr << "Starting training from scratch." << std::endl;
    }
  } else {
    if (inference_mode) {
      std::cerr << "Weights file '" << weights_filename
                << "' not found. Cannot run inference. Exiting." << std::endl;
      return 1;
    }
    std::cout << "No existing weights file found or loading disabled. Starting "
                 "training from scratch."
              << std::endl;
  }
  // End Load Weights

  if (!inference_mode) {
    // Training Mode
    Adam optimizer(learning_rate);
    CrossEntropyLoss criterion;

    std::cout << "Starting Transformer training..." << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
      optimizer.zero_grad();

      std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> batch_data =
          data_loader.get_batch();
      std::shared_ptr<Tensor> encoder_input_ids = batch_data.first;
      std::shared_ptr<Tensor> target_output_ids = batch_data.second;
      std::shared_ptr<Tensor> decoder_input_ids = encoder_input_ids;

      // Forward pass
      std::shared_ptr<Tensor> logits =
          model.forward(encoder_input_ids, decoder_input_ids, true);

      // Reshape logits
      std::shared_ptr<Tensor> reshaped_logits =
          logits->reshape({batch_size * decoder_seq_length, target_vocab_size});

      // Compute loss
      std::shared_ptr<Tensor> loss =
          criterion.compute_loss(reshaped_logits, target_output_ids);

      std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                << "], Loss: " << vector_to_string(loss->get_data())
                << std::endl;

      // Backward pass
      criterion.backward(loss);
      optimizer.step();
    }

    std::cout << "\nTransformer training finished." << std::endl;

    // Save weights after training
    try {
      std::cout << "Saving final weights to " << weights_filename << "..."
                << std::endl;
      model.save_weights(weights_filename);
      std::cout << "Weights saved successfully." << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Failed to save weights: " << e.what() << std::endl;
    }
    // End Training Mode
  } else {
    // Inference Mode
    if (!weights_loaded) {
      std::cerr << "Weights were not loaded (or failed to load). Cannot run "
                   "inference. Exiting."
                << std::endl;
      return 1;
    }

    std::cout << "\n--- Running Inference ---" << std::endl;
    std::cout << "Initial prompt: \"" << initial_prompt << "\"" << std::endl;

    int current_seq_len = input_seq_length;

    // Tokenize the initial prompt
    std::shared_ptr<Tensor> current_input_ids =
        string_to_tensor(initial_prompt, data_loader, current_seq_len);
    std::vector<float> generated_ids = current_input_ids->get_data();

    // Remove padding from initial prompt display if any was added by
    // string_to_tensor
    while (!generated_ids.empty() && generated_ids.back() == 0.0f) {
      generated_ids.pop_back();
    }
    // Get actual length of the prompt after removing padding
    int current_total_len = generated_ids.size();

    std::cout << "Generating..." << std::endl;
    for (float id_float : generated_ids) {
      std::cout << data_loader.get_char_from_id(static_cast<int>(id_float));
    }
    std::cout << std::flush;

    for (int i = 0; i < max_generate_length; ++i) {
      std::vector<float> step_input_vec;
      int start_idx = std::max(0, current_total_len - current_seq_len);
      for (int k = start_idx; k < current_total_len; ++k) {
        step_input_vec.push_back(generated_ids[k]);
      }
      // Pad if the current sequence is shorter than model's input length
      while (step_input_vec.size() < current_seq_len) {
        step_input_vec.push_back(pad_token_id);
      }

      // Create tensor for this step (batch size 1)
      std::shared_ptr<Tensor> step_input_tensor =
          Tensor::create({1, current_seq_len},
                         std::make_shared<std::vector<float>>(step_input_vec));

      // Encoder input and Decoder input are the same for autoregressive
      // generation
      std::shared_ptr<Tensor> encoder_input = step_input_tensor;
      std::shared_ptr<Tensor> decoder_input = step_input_tensor;

      // Forward pass (is_training = false)
      std::shared_ptr<Tensor> logits =
          model.forward(encoder_input, decoder_input, false);
      // Logits shape: (1, current_seq_len, target_vocab_size)

      int last_token_index = std::min(current_total_len, current_seq_len) - 1;
      if (last_token_index < 0) {
        std::cerr << "\nError: last_token_index is negative ("
                  << last_token_index
                  << "). current_total_len=" << current_total_len
                  << ". Breaking." << std::endl;
        break;
      }

      const auto &logits_data = logits->get_data();
      size_t logits_offset = last_token_index * target_vocab_size;

      // Temperature sampling for more diverse outputs
      float temperature = 1.0f;
      std::vector<float> probs(target_vocab_size);
      float max_logit = -std::numeric_limits<float>::infinity();

      if (logits_offset + target_vocab_size > logits_data.size()) {
        std::cerr << "\nError: Logits offset (" << logits_offset
                  << ") + vocab_size (" << target_vocab_size
                  << ") exceeds logits_data size (" << logits_data.size()
                  << "). last_token_index=" << last_token_index << ". Breaking."
                  << std::endl;
        break;
      }

      // Find max for numerical stability
      for (int j = 0; j < target_vocab_size; ++j) {
        if (logits_data[logits_offset + j] > max_logit) {
          max_logit = logits_data[logits_offset + j];
        }
      }

      // Compute softmax with temperature
      float sum_exp = 0.0f;
      for (int j = 0; j < target_vocab_size; ++j) {
        probs[j] = std::exp((logits_data[logits_offset + j] - max_logit) /
                            temperature);
        sum_exp += probs[j];
      }
      for (int j = 0; j < target_vocab_size; ++j) {
        probs[j] /= sum_exp;
      }

      // Sample from the distribution
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      float random_val = dist(global_rng);
      float cumsum = 0.0f;
      int predicted_id = 0;
      for (int j = 0; j < target_vocab_size; ++j) {
        cumsum += probs[j];
        if (random_val <= cumsum) {
          predicted_id = j;
          break;
        }
      }

      // Append the predicted ID to our sequence
      generated_ids.push_back(static_cast<float>(predicted_id));
      current_total_len++;

      // Convert the predicted ID to a character and print it
      char next_char = data_loader.get_char_from_id(predicted_id);

      std::cout << next_char << std::flush;
    }

    std::cout << "\n--- Generation Complete ---" << std::endl;
  }

  return 0;
}