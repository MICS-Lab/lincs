#include "lib.hpp"

int main() {


Foo* foo = makeFoo();

for (int i = 0; i != 1'000'000'000; ++i) {
  foo->no_virtual();
}

}
