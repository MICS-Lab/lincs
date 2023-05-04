from __future__ import annotations
import shutil

import subprocess
import os
import textwrap


def main():
    # With plad NOT installed
    #########################

    subprocess.run([f"pip3", "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL, check=True)

    # Install plad
    ##############

    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("plad.egg-info", ignore_errors=True)
    subprocess.run([f"pip3", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    # With plad installed
    #####################

    os.chdir(os.path.expanduser("~"))

    print("Use as a standalone command-line tool")
    domain = subprocess.run(
        ["plad", "generate", "classification-domain", "3", "2", "-"],
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    print(domain)
    subprocess.run(
        ["plad", "generate", "classification-model", "-", "-"],
        check=True,
        input=domain,
        universal_newlines=True,
    )
    print()
    print("Use as an executable Python module")
    subprocess.run(["python3", "-m", "plad", "generate", "classification-domain", "4", "3"], check=True)
    print()
    print("Use as a Python package")
    subprocess.run(
        ["python3"],
        check=True,
        # @todo Reduce this example, turn its tests into unit tests
        input=textwrap.dedent("""
            import io
            import sys
            import plad

            criterion = plad.Criterion('Physic grade', plad.ValueType.real, plad.CategoryCorrelation.growing)
            print(criterion.name, criterion.value_type, criterion.category_correlation)
            criterion.name = 'Physics grade'
            criterion.value_type = plad.ValueType.real
            criterion.category_correlation = plad.CategoryCorrelation.growing

            domain = plad.Domain(
                [
                    criterion,
                    plad.Criterion('Literature grade', plad.ValueType.real, plad.CategoryCorrelation.growing),
                ],
                (
                    plad.Category('Bad'),
                    plad.Category('Good'),
                ),
            )

            criterion = domain.criteria[0]
            print(criterion.name, criterion.value_type, criterion.category_correlation)
            print(domain.categories[0].name)
            domain.categories[0].name = "Terrible"
            domain.categories[1].name = "Ok, I guess"

            buf = io.StringIO()
            domain.dump(buf)
            print(buf.getvalue())

            model = plad.Model(domain, [plad.Boundary([10.,10.], plad.SufficientCoalitions(plad.SufficientCoalitionsKind.weights, [0.4, 0.7]))])
            model.dump(sys.stdout)
            print()
        """),
        universal_newlines=True,
    )
    print()
    print("Use as a C++ library")
    subprocess.run(
        [
            "g++",
            "-x", "c++", "-",
            "-I/home/user/.local/lib/python3.10/site-packages/plad/libplad",
            "-L/home/user/.local/lib/python3.10/site-packages", "-lplad.cpython-310-x86_64-linux-gnu",
        ],
        check=True,
        # @todo Reduce this example, turn its tests into unit tests
        input=textwrap.dedent("""
            #include <plad.hpp>

            #include <iostream>
            #include <sstream>

            int main() {
                plad::Domain domain{
                    {
                        {"Literature grade", plad::Domain::Criterion::ValueType::real, plad::Domain::Criterion::CategoryCorrelation::growing},
                        {"Physics grade", plad::Domain::Criterion::ValueType::real, plad::Domain::Criterion::CategoryCorrelation::growing},
                    },
                    {
                        {"Fail"},
                        {"Pass"},
                    }
                };

                domain.dump(std::cout);
                std::cout << std::endl;

                plad::Model model{&domain, {{{10.f, 10.f}, {plad::Model::SufficientCoalitions::Kind::weights, {0.4f, 0.7f}}}}};
                {
                    std::ostringstream oss;
                    model.dump(oss);
                    std::cout << oss.str() << std::endl;
                    std::istringstream iss(oss.str());
                    plad::Model model2 = plad::Model::load(&domain, iss);

                    model2.dump(std::cout);
                    std::cout << std::endl;
                }

                plad::AlternativesSet alternatives{&domain, {{"Alice", {11.f, 12.f}, "Pass"}, {"Bob", {9.f, 11.f}, "Fail"}}};
                {
                    std::ostringstream oss;
                    alternatives.dump(oss);
                    std::cout << oss.str();
                    std::istringstream iss(oss.str());
                    plad::AlternativesSet alternatives2 = plad::AlternativesSet::load(&domain, iss);

                    alternatives2.dump(std::cout);
                }
            }
        """),
        universal_newlines=True,
    )
    subprocess.run(["./a.out"], check=True, env={"LD_LIBRARY_PATH": "/home/user/.local/lib/python3.10/site-packages"})

    with open("/wd/plad help-all.txt", "w") as f:
        subprocess.run(["plad", "help-all"], stdout=f, check=True)


if __name__ == "__main__":
    main()
