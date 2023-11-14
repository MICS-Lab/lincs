class Foo {
 public:
  virtual void yes_virtual() = 0;
  void no_virtual();
};

class ActualFoo : public Foo {
 public:
  void yes_virtual() override;
};

Foo* makeFoo();
