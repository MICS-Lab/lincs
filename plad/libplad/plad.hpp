#include <ostream>
#include <string>


namespace plad {

class Domain {
 public:
  Domain(int criteria_count_, int categories_count_): criteria_count(criteria_count_), categories_count(categories_count_) {}

  void dump(std::ostream&) const;
  static Domain load(std::istream&);

 private:
  int criteria_count;
  int categories_count;
};

}  // namespace plad
