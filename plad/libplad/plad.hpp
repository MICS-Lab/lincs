#include <ostream>
#include <string>
#include <vector>


namespace plad {

class Domain {
 public:
  struct Criterion {
    enum class ValueType {
      real,
    };

    enum class CategoryCorrelation {
      growing,
    };

    std::string name;
    ValueType value_type;
    CategoryCorrelation category_correlation;

    // @todo Remove these constructors:
    // The struct is usable without them in C++, and they were added only to allow using bp::init in the Python module
    Criterion() {}
    Criterion(const std::string& name_, ValueType value_type_, CategoryCorrelation category_correlation_): name(name_), value_type(value_type_), category_correlation(category_correlation_) {}
  };

  struct Category {
    std::string name;

    // @todo Remove these constructors (see Criterion)
    Category() {}
    Category(const std::string& name_): name(name_) {}
  };

 public:
  Domain(const std::vector<Criterion>& criteria_, const std::vector<Category>& categories_): criteria(criteria_), categories(categories_) {}

  void dump(std::ostream&) const;
  static Domain load(std::istream&);

 private:
  std::vector<Criterion> criteria;
  std::vector<Category> categories;
};

}  // namespace plad
