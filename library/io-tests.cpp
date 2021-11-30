// Copyright 2021 Vincent Jacques

#include "io.hpp"
#include "test-utils.hpp"


using ppl::io::Model;

TEST(SerializeDeserialize, Model) {
  Model model(3, 2, {{0.3, 0.4, 0.5}}, {0.7, 0.8, 0.9});

  std::ostringstream oss;
  model.save_to(oss);
  auto serialized = oss.str();

  EXPECT_EQ(
    serialized,
    "3\n"
    "2\n"
    "0.291667 0.333333 0.375\n"
    "0.416667\n"
    "0.3 0.4 0.5\n");

  std::istringstream iss(serialized);
  Model deserialized = Model::load_from(iss);

  EXPECT_EQ(deserialized.criteria_count, 3);
  EXPECT_EQ(deserialized.categories_count, 2);
  EXPECT_NEAR(deserialized.profiles[0][0], 0.3, 1e-5);
  EXPECT_NEAR(deserialized.profiles[0][1], 0.4, 1e-5);
  EXPECT_NEAR(deserialized.profiles[0][2], 0.5, 1e-5);
  EXPECT_NEAR(deserialized.weights[0], 0.7, 1e-5);
  EXPECT_NEAR(deserialized.weights[1], 0.8, 1e-5);
  EXPECT_NEAR(deserialized.weights[2], 0.9, 1e-5);
}
