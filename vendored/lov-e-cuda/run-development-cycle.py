#!/usr/bin/env python3

import multiprocessing
import re
import socket
import subprocess


def main():
    regenerate(
        "lov-e.hpp",
        re.compile(r"// (BEGIN|END) GENERATED SECTION: (.*)"),
        {class_.key: class_ for class_ in [ArraysAndArrayViewSection]},
    )
    targets = ["default"]
    if socket.gethostname() in ["sam"]:
        targets.append("memcheck-tests")
    subprocess.run(["./make.sh", f"-j{multiprocessing.cpu_count()}"] + targets, check=True)
    subprocess.run(["./make.sh", "-j1", "examples"], check=True)
    regenerate(
        "README.md",
        re.compile(r"<!-- (BEGIN|END) GENERATED SECTION: (.*) -->"),
        {class_.key: class_ for class_ in [ExamplesPerformanceTableSection, UserManualSnippetSection]},
    )


def regenerate(file_name, generated_re, section_classes):
    with open(file_name) as f:
        original_lines = [line.rstrip() for line in f.readlines()]

    def gen():
        sections = [HandWrittenSection()]
        for line in original_lines:
            match = generated_re.fullmatch(line)
            if match:
                if match.group(1) == "BEGIN":
                    name = match.group(2)
                    sub_match = re.fullmatch(r"(.*)\((.*)\)", name)
                    if sub_match:
                        section_class_name = sub_match.group(1)
                        section_arguments = (sub_match.group(2),)
                    else:
                        section_class_name = name
                        section_arguments = ()
                    sections.append(section_classes[section_class_name](*section_arguments))
                    sections[-1].entry = match.group(0)
                elif match.group(1) == "END":
                    sections[-1].exit = match.group(0)
                    sections.append(HandWrittenSection())
                else:
                    assert False
            else:
                sections[-1].record_original_line(line)

        for section in sections:
            if section.key is not None:
                yield section.entry
                yield ""
            for line in section.generate():
                yield line
            if section.key is not None:
                yield ""
                yield section.exit

    new_lines = list(gen())
    if new_lines != original_lines:
        with open(file_name, "w") as f:
            for line in new_lines:
                f.write(f"{line}\n")


class HandWrittenSection:
    key = None

    def __init__(self):
        self.__original_lines = []

    def record_original_line(self, line):
        self.__original_lines.append(line)

    def generate(self):
        return self.__original_lines


class ExamplesPerformanceTableSection:
    key = "examples-performance-table"

    def record_original_line(self, _):
        pass

    def generate(self):
        yield "| Example | Without *Lov-e-cuda* | With *Lov-e-cuda* |"
        yield "| --- | --- | --- |"

        # @todo Average a few executions (10?) to get more significant numbers

        def parse_mandelbrot(base_name):
            with open(f"build/release/examples/{base_name}.log") as f:
                (line,) = f.readlines()
                m = re.fullmatch(r"Mandelbrot set computed in (.*) s, at (.*) Mpix/s", line.rstrip())
                return f"{int(float(m.group(1)) * 1000)} ms *i.e.* {int(float(m.group(2)))} Mpix/s"

        for (description, name, parse) in [
            ("Mandelbrot<br>(static parallelism)", "mandelbrot", parse_mandelbrot),
            ("Mandelbrot<br>(dynamic parallelism)", "mandelbrot-dyn", parse_mandelbrot)
        ]:
            yield f"| {description} | {parse(name)} | {parse(name + '-lov-e')} |"


class UserManualSnippetSection:
    key = "user-manual-snippet"

    def __init__(self, snippet_name):
        self.__snippet_name = snippet_name

    def record_original_line(self, _):
        pass

    def generate(self):
        do_yield = False
        with open("tests/user-manual.cu") as f:
            for line in f:
                line = line.rstrip()
                if line.lstrip() == f"// BEGIN {self.__snippet_name}":
                    indent = len(line) - len(line.lstrip())
                    do_yield = True
                elif line.lstrip() == f"// END {self.__snippet_name}":
                    break
                elif do_yield:
                    line = line[indent:]
                    if line:
                        yield "    " + line
                    else:
                        yield ""
        assert do_yield


class ArraysAndArrayViewSection:
    key = "arrays-and-array-views"

    def record_original_line(self, _):
        pass

    def generate(self):
        first = True
        for n in range(1, 6):
            if first:
                first = False
            else:
                yield ""
            yield from self.generate_one(n)

    def generate_one(self, n):
        ds = list(range(n - 1, -1, -1))
        lower_ds = ds[1:]

        def sep(f, sep=', ', ds=ds):
            return sep.join(f.format(d=d) for d in ds)

        def generate_array_view(where="Where"):
            if n == 1:
                yield "template<typename T>"
                yield f"class ArrayView{n}D<{where}, T> {{"
            else:
                yield "template<typename Where, typename T>"
                yield f"class ArrayView{n}D {{"
            yield " public:"
            yield "  // Constructor"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n}D({sep('std::size_t s{d}')}, T* data) :"
            yield f"    {sep('_s{d}(s{d})')}, _data(data) {{}}"
            yield ""
            yield '  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)'
            yield ""
            yield "  // Generalized copy constructor and operator"
            yield "  template<typename U>"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n}D(const ArrayView{n}D<{where}, U>& o) :"
            yield f"    {sep('_s{d}(o.s{d}())')}, _data(o.data()) {{}}"
            yield ""
            yield "  template<typename U>"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n}D& operator=(const ArrayView{n}D<{where}, U>& o) {{"
            for d in ds:
                yield f"    _s{d} = o.s{d}();"
            yield "    _data = o.data();"
            yield "    return *this;"
            yield "  }"
            yield ""
            yield "  // Generalized conversion operator"
            yield "  template<typename U>"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  operator ArrayView{n}D<Anywhere, U>() {{"
            yield f"    return ArrayView{n}D<Anywhere, U>({sep('_s{d}')}, _data);"
            yield "  }"
            yield ""
            yield "  // Accessors"
            for d in ds:
                yield "  HOST_DEVICE_DECORATORS"
                yield f"  std::size_t s{d}() const {{ return _s{d}; }}"
            yield ""
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  std::size_t total_size() const {{ return {sep('_s{d}', ' * ')}; }}"
            yield ""
            if where in ["Anywhere", "Where"]:
                yield "  HOST_DEVICE_DECORATORS"
            if where == "Device":
                yield "#ifdef __NVCC__"
                yield "  __device__"
            if n == 1:
                yield f"  T& operator[](unsigned i{n-1}) const {{"
            else:
                yield f"  ArrayView{n-1}D<{where}, T> operator[](unsigned i{n-1}) const {{"
            yield f"    assert(i{n-1} < _s{n-1});"
            if n == 1:
                yield "    return *(_data + i0);"
            else:
                yield f"    return ArrayView{n-1}D<{where}, T>({sep('_s{d}', ', ', lower_ds)}, _data + i{n-1} * {sep('_s{d}', ' * ', lower_ds)});"
            yield "  }"
            if where == "Device":
                yield "#endif"
            yield ""
            yield "  HOST_DEVICE_DECORATORS"
            yield "  T* data() const { return _data; }"
            if where != "Anywhere":
                yield ""
                yield "  void fill_with_zeros() const {"
                yield f"    {where}::template memreset<T>(total_size(), data());"
                yield "  }"
                yield ""
                yield "  // Clonable"
                yield "  template<typename WhereTo>"
                yield f"  Array{n}D<WhereTo, typename std::remove_const<T>::type> clone_to() const;"
                yield ""
                yield f"  Array{n}D<{where}, typename std::remove_const<T>::type> clone() const {{ return clone_to<{where}>(); }}"
            yield ""
            yield " private:"
            for d in ds:
                yield f"  std::size_t _s{d};"
            yield "  T* _data;"
            if where != "Anywhere":
                yield ""
                yield f"  friend class Array{n}D<{where}, typename std::remove_const<T>::type>;"
            yield "};"

        def generate_array():
            yield "template<typename WhereFrom, typename WhereTo, typename T>"
            yield f"void copy(ArrayView{n}D<WhereFrom, T> src, ArrayView{n}D<WhereTo, typename std::remove_const<T>::type> dst) {{"
            for d in ds:
                yield f"  assert(dst.s{d}() == src.s{d}());"
            yield ""
            yield "  From<WhereFrom>::template To<WhereTo>::template copy("
            yield f"    {sep('src.s{d}()', ' * ')}, src.data(), dst.data());"
            yield "}"

            yield ""

            yield "template<typename Where, typename T>"
            yield f"class Array{n}D : public ArrayView{n}D<Where, const T> {{"
            yield " public:"
            yield "  // RAII"
            yield "  template<typename W = Where, typename = typename std::enable_if<!W::can_be_allocated_on_device>::type>"
            yield f"  Array{n}D({sep('std::size_t s{d}')}, Uninitialized) :"
            yield f"    ArrayView{n}D<Where, const T>({sep('s{d}')}, Where::template alloc<T>({sep('s{d}', ' * ')}))"
            yield "  {}"
            yield ""
            yield "  template<"
            yield "    typename = void,"
            yield "    typename W = Where,"
            yield "    typename = typename std::enable_if<W::can_be_allocated_on_device>::type"
            yield "  >"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  Array{n}D({sep('std::size_t s{d}')}, Uninitialized) :"
            yield f"    ArrayView{n}D<Where, const T>({sep('s{d}')}, Where::template alloc<T>({sep('s{d}', ' * ')}))"
            yield "  {}"
            yield ""
            yield f"  Array{n}D({sep('std::size_t s{d}')}, Zeroed) :"
            yield f"    ArrayView{n}D<Where, const T>({sep('s{d}')}, Where::template alloc_zeroed<T>({sep('s{d}', ' * ')}))"
            yield "  {}"
            yield ""
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ~Array{n}D() {{"
            yield "    free();"
            yield "  }"
            yield ""
            yield "  // Accessors"
            yield "  HOST_DEVICE_DECORATORS"
            yield "  T* data() const { return const_cast<T*>(this->_data); }"
            yield ""
            if n == 1:
                yield "  // @todo Fix decorators: forbid dereferencing host memory on host and vice versa"
            yield "  HOST_DEVICE_DECORATORS"
            if n == 1:
                yield "  T& operator[](unsigned i0) const {"
            else:
                yield f"  ArrayView{n-1}D<Where, T> operator[](unsigned i{n-1}) const {{"
            yield f"    assert(i{n-1} < this->_s{n-1});"
            if n == 1:
                yield "    return *(data() + i0);"
            else:
                yield f"    return ArrayView{n-1}D<Where, T>("
                yield f"      {sep('this->_s{d}', ', ', lower_ds)}, data() + i{n-1} * {sep('this->_s{d}', ' * ', lower_ds)});"
            yield "  }"
            yield ""
            yield "  void fill_with_zeros() const {"
            yield "    Where::template memreset<T>(this->total_size(), data());"
            yield "  }"
            yield ""
            yield "  // Not copyable"
            yield f"  Array{n}D(const Array{n}D&) = delete;"
            yield f"  Array{n}D& operator=(const Array{n}D&) = delete;"
            yield ""
            yield "  // But movable"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  Array{n}D(Array{n}D&& o) : ArrayView{n}D<Where, const T>(o) {{"
            for d in ds:
                yield f"    o._s{d} = 0;"
            yield "    o._data = nullptr;"
            yield "  }"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  Array{n}D& operator=(Array{n}D&& o) {{"
            yield "    free();"
            yield f"    static_cast<ArrayView{n}D<Where, const T>&>(*this) = o;"
            for d in ds:
                yield f"    o._s{d} = 0;"
            yield "    o._data = nullptr;"
            yield "    return *this;"
            yield "  }"
            yield ""
            yield " private:"
            yield "  HOST_DEVICE_DECORATORS"
            yield "  void free() {"
            yield "    Where::free(data());"
            yield "  }"
            yield "};"

            yield ""

            yield "template<typename Where, typename T>"
            yield "HOST_DEVICE_DECORATORS"
            yield f"ArrayView{n}D<Where, T> ref(const Array{n}D<Where, T>& a) {{"
            yield f"  return ArrayView{n}D<Where, T>({sep('a.s{d}()')}, a.data());"
            yield "}"

            yield ""

            yield "template<typename Where, typename T>"
            yield "HOST_DEVICE_DECORATORS"
            yield f"ArrayView{n}D<Where, T> ref(const ArrayView{n}D<Where, T>& a) {{"
            yield f"  return a;"
            yield "}"

            yield ""

            if n == 1:
                yield "template<typename T>"
                yield "template<typename WhereTo>"
                yield f"Array{n}D<WhereTo, typename std::remove_const<T>::type> ArrayView{n}D<Host, T>::clone_to() const {{"
                yield f"  Array{n}D<WhereTo, typename std::remove_const<T>::type> dst(this->s0(), uninitialized);"
                yield "  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)"
                yield "  return dst;"
                yield "}"
                yield ""
                yield "template<typename T>"
                yield "template<typename WhereTo>"
                yield f"Array{n}D<WhereTo, typename std::remove_const<T>::type> ArrayView{n}D<Device, T>::clone_to() const {{"
                yield f"  Array{n}D<WhereTo, typename std::remove_const<T>::type> dst(this->s0(), uninitialized);"
                yield "  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)"
                yield "  return dst;"
                yield "}"
            else:
                yield "template<typename WhereFrom, typename T>"
                yield "template<typename WhereTo>"
                yield f"Array{n}D<WhereTo, typename std::remove_const<T>::type> ArrayView{n}D<WhereFrom, T>::clone_to() const {{"
                yield f"  Array{n}D<WhereTo, typename std::remove_const<T>::type> dst("
                yield f"    {sep('this->s{d}()')}, uninitialized);"
                yield "  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)"
                yield "  return dst;"
                yield "}"

        yield f"template<typename Where, typename T> class Array{n}D;"
        yield ""
        if n == 1:
            yield "template<typename Where, typename T> class ArrayView1D;"
            yield ""
            yield from generate_array_view("Anywhere")
            yield ""
            yield from generate_array_view("Host")
            yield ""
            yield from generate_array_view("Device")
        else:
            yield from generate_array_view()
        yield ""
        yield from generate_array()



if __name__ == "__main__":
    main()
