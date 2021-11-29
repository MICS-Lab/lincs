// Copyright 2021 Vincent Jacques

#include <iostream>
#include <sstream>
#include <fstream>

#include "../library/generate.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " NB_CRIT NB_CAT SEED\n"
    "\n"
    "Generate a pseudo-random model with NB_CRIT criteria and NB_CAT categories,\n"
    "with random seed SEED.\n"
    "\n"
    "The model will be stored in file 'model-NB_CRIT-NB_CAT-SEED.txt'.\n"
    "\n"
    "Model generation is deterministic: the same NB_CRIT, NB_CAT and SEED will\n"
    "always generate the same file."
    << std::endl;
  exit(1);
}

int main(int argc, char* argv[]) {
  if (argc != 4) usage(argv[0]);

  unsigned int criteria_count;
  unsigned int categories_count;
  unsigned int seed;

  try {
    criteria_count = std::stoi(argv[1]);
    categories_count = std::stoi(argv[2]);
    seed = std::stoi(argv[3]);
  } catch (std::invalid_argument) {
    usage(argv[0]);
  }

  std::mt19937 gen(seed);
  auto model = ppl::generate::model(&gen, criteria_count, categories_count);

  std::ostringstream file_name;
  file_name << "model-" << criteria_count << "-" << categories_count << "-" << seed << ".txt";
  std::ofstream file(file_name.str());
  model.save_to(file);
}
