// Copyright 2021 Vincent Jacques

#include <iostream>
#include <fstream>

#include <CLI11.hpp>

#include "../library/generate.hpp"


int main(int argc, char* argv[]) {
  CLI::App app(
    "Generate a pseudo-random model with NB_CRIT criteria and NB_CAT categories,\n"
    "from random seed SEED.\n"
    "\n"
    "The generated model is printed on standard output.\n"
    "\n"
    "Model generation is deterministic: the same NB_CRIT, NB_CAT and SEED\n"
    "always generate the same model.\n");

  unsigned int criteria_count;
  app.add_option("NB_CRIT", criteria_count)->required();

  unsigned int categories_count;
  app.add_option("NB_CAT", categories_count)->required();

  unsigned int seed;
  app.add_option("SEED", seed)->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  std::mt19937 gen(seed);
  ppl::generate::model(&gen, criteria_count, categories_count).save_to(std::cout);
}
