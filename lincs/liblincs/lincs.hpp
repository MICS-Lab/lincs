#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>


namespace lincs {

struct Domain {
  struct Criterion {
    std::string name;

    enum class ValueType {
      real,
      // @todo Add integer
      // @todo Add enumerated
    } value_type;

    enum class CategoryCorrelation {
      growing,
      // @todo Add decreasing
      // @todo Add single-peaked
      // @todo Add single-valleyed
      // @todo Add unknown
    } category_correlation;

    // @todo Remove these constructors
    // The struct is usable without them in C++, and they were added only to allow using bp::init in the Python module
    // (Do it for other structs as well)
    Criterion() {}
    Criterion(const std::string& name_, ValueType value_type_, CategoryCorrelation category_correlation_): name(name_), value_type(value_type_), category_correlation(category_correlation_) {}

    // @todo Remove this operator
    // The struct is usable without it in C++, and it was added only to allow using bp::vector_indexing_suite in the Python module
    // (Do it for other structs as well)
    bool operator==(const Criterion& other) const {
      return name == other.name && value_type == other.value_type && category_correlation == other.category_correlation;
    }
  };

  std::vector<Criterion> criteria;

  struct Category {
    std::string name;

    Category() {}
    Category(const std::string& name_): name(name_) {}

    bool operator==(const Category& other) const { return name == other.name; }
  };

  std::vector<Category> categories;

  Domain(const std::vector<Criterion>& criteria_, const std::vector<Category>& categories_): criteria(criteria_), categories(categories_) {}

  void dump(std::ostream&) const;
  static Domain load(std::istream&);

  static Domain generate(unsigned criteria_count, unsigned categories_count, unsigned random_seed);
};

struct Model {
  const Domain& domain;

  struct SufficientCoalitions {
    // Sufficient coalitions form an https://en.wikipedia.org/wiki/Upper_set in the set of parts of the set of criteria.
    // This upset can be defined:
    enum class Kind {
      weights,  // by the weights of the criteria
      // @todo Add upset_roots,  // explicitly by its roots
    } kind;

    std::vector<float> criterion_weights;

    SufficientCoalitions() {};
    SufficientCoalitions(Kind kind_, const std::vector<float>& criterion_weights_): kind(kind_), criterion_weights(criterion_weights_) {}
  };

  struct Boundary {
    std::vector<float> profile;
    SufficientCoalitions sufficient_coalitions;

    Boundary() {};
    Boundary(const std::vector<float>& profile_, const SufficientCoalitions& sufficient_coalitions_): profile(profile_), sufficient_coalitions(sufficient_coalitions_) {}

    bool operator==(const Boundary& other) const { return profile == other.profile && sufficient_coalitions.kind == other.sufficient_coalitions.kind && sufficient_coalitions.criterion_weights == other.sufficient_coalitions.criterion_weights; }
  };

  std::vector<Boundary> boundaries;  // boundary_index 0 is between category_index 0 and category_index 1

  Model(const Domain& domain_, const std::vector<Boundary>& boundaries_) : domain(domain_), boundaries(boundaries_) {}

  void dump(std::ostream&) const;
  static Model load(const Domain&, std::istream&);

  static Model generate_mrsort(const Domain&, unsigned random_seed, std::optional<float> fixed_weights_sum = std::nullopt);
};

struct Alternative {
  std::string name;
  std::vector<float> profile;
  std::optional<std::string> category;

  Alternative() {}
  Alternative(const std::string& name_, const std::vector<float>& profile_, const std::optional<std::string>& category_): name(name_), profile(profile_), category(category_) {}

  bool operator==(const Alternative& other) const { return name == other.name && profile == other.profile && category == other.category; }
};

class BalancedAlternativesGenerationException : public std::exception {
 public:
  explicit BalancedAlternativesGenerationException(const std::map<std::string, unsigned>& histogram_) : histogram(histogram_) {}

  const char* what() const noexcept override {
    return "Unable to generate balanced alternatives. Try increasing the allowed imbalance, or use a more lenient model?";
  }

  std::map<std::string, unsigned> histogram;
};

struct Alternatives {
  const Domain& domain;
  std::vector<Alternative> alternatives;

  Alternatives(const Domain& domain_, const std::vector<Alternative>& alternatives_): domain(domain_), alternatives(alternatives_) {}

  void dump(std::ostream&) const;
  static Alternatives load(const Domain&, std::istream&);

  static Alternatives generate(
    const Domain&,
    const Model&,
    unsigned alternatives_count,
    unsigned random_seed,
    std::optional<float> max_imbalance = std::nullopt
  );
};

struct ClassificationResult {
  unsigned unchanged;
  unsigned changed;
};

ClassificationResult classify_alternatives(const Domain&, const Model&, Alternatives*);

}  // namespace lincs
