#include "lib.hpp"


void Foo::no_virtual() {}
void ActualFoo::yes_virtual() {}

Foo* makeFoo() { return new ActualFoo; }
