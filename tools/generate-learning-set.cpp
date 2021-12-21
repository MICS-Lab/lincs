// Copyright 2021 Vincent Jacques

#include <iostream>
#include <fstream>

#include <chrones.hpp>
#include <CLI11.hpp>

#include "../library/generate.hpp"


CHRONABLE("generate-learning-set")

int main(int argc, char* argv[]) {
  CHRONE();

  CLI::App app(
    "Generate a pseudo-random learning set of NB_ALT alternatives,\n"
    "for the model taken from file MODEL.txt, and from random seed SEED.\n"
    "\n"
    "The generated learning set is printed on standard output.\n"
    "\n"
    "Learning set generation is deterministic: the same MODEL.txt file, NB_ALT and\n"
    "SEED always generate the same learning set.\n");

  std::string model_file_name;
  app.add_option("MODEL.txt", model_file_name)
    ->required()
    ->check(CLI::ExistingFile);

  unsigned int alternatives_count;
  app.add_option("NB_ALT", alternatives_count)->required();

  unsigned int seed;
  app.add_option("SEED", seed)->required();

  CLI11_PARSE(app, argc, argv);

  std::ifstream model_file(model_file_name);
  auto model = ppl::io::Model::load_from(model_file);

  std::mt19937 gen(seed);
  ppl::generate::learning_set(&gen, model, alternatives_count).save_to(std::cout);
}
