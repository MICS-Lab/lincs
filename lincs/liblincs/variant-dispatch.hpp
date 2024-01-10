// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__VARIANT_DISPATCH_HPP
#define LINCS__VARIANT_DISPATCH_HPP

#include <variant>


namespace lincs {

// Inspired by https://stackoverflow.com/q/45707857/905845

template<class... Ts>
struct dispatcher : Ts... {
  template<class> static inline constexpr bool always_false_v = false;

  template<typename T>
  auto operator()(T const&) const {
    static_assert(always_false_v<T>, "A variant type is not dispatched");
  }

  using Ts::operator()...;
};

template<class... Ts>
dispatcher(Ts...) -> dispatcher<Ts...>;

template<class U, class... Ts>
auto dispatch(const U& u, const Ts... ts) {
  return std::visit(dispatcher{ts...}, u);
}

}  // namespace lincs

#endif  // LINCS__VARIANT_DISPATCH_HPP
