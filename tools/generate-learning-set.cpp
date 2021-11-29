// Copyright 2021 Vincent Jacques

#include <iostream>
#include <sstream>
#include <fstream>

#include "../library/generate.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " MODEL.txt NB_ALT SEED\n"
    "\n"
    "Generate a pseudo-random learning set for the model taken from file MODEL.txt\n"
    "with NB_ALT alternatives and random seed SEED.\n"
    "\n"
    "The learning set will be stored in file 'learning-set--MODEL--NB_ALT-SEED.txt'.\n"
    "\n"
    "Learning set generation is deterministic: the same MODEL.txt file, NB_ALT and\n"
    "SEED will always generate the same file."
    << std::endl;
  exit(1);
}

int main(int argc, char* argv[]) {
  if (argc != 4) usage(argv[0]);

  const std::string model_file_name(argv[1]);
  std::ifstream model_file(model_file_name);
  auto model = ppl::io::Model::load_from(model_file);
  unsigned int alternatives_count;
  unsigned int seed;

  try {
    alternatives_count = std::stoi(argv[2]);
    seed = std::stoi(argv[3]);
  } catch (std::invalid_argument) {
    usage(argv[0]);
  }

  std::mt19937 gen(seed);
  auto learning_set = ppl::generate::learning_set(&gen, model, alternatives_count);

  std::ostringstream file_name;
  file_name
    << "learning-set--" << model_file_name.substr(0, model_file_name.size() - 4)
    << "--" << alternatives_count << "-" << seed << ".txt";
  std::ofstream file(file_name.str());
  learning_set.save_to(file);
}
