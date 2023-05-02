#include "plad.hpp"

#include <yaml-cpp/yaml.h>


namespace plad {

void Domain::dump(std::ostream& os) const {
  YAML::Emitter out;

  out << YAML::BeginMap;
  out << YAML::Key << "criteria";
  out << YAML::Value << criteria_count;
  out << YAML::Key << "categories";
  out << YAML::Value << categories_count;
  out << YAML::EndMap;

  os << out.c_str();
}

Domain Domain::load(std::istream& is) {
  YAML::Node domain = YAML::Load(is);

  return Domain(
    domain["criteria"].as<int>(),
    domain["categories"].as<int>()
  );
}

}  // namespace plad
