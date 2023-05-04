from __future__ import annotations
import shutil

import subprocess
import os
import textwrap


def main():
    # With lincs NOT installed
    ##########################

    subprocess.run([f"pip3", "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL, check=True)

    # Install lincs
    ###############

    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)
    subprocess.run([f"pip3", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    # With lincs installed
    ######################

    os.chdir(os.path.expanduser("~"))

    print("Use as a standalone command-line tool")
    domain = subprocess.run(
        ["lincs", "generate", "classification-domain", "3", "2"],
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    print(domain)
    subprocess.run(
        ["lincs", "generate", "classification-model", "-"],
        check=True,
        input=domain,
        universal_newlines=True,
    )
    print()
    print("Use as an executable Python module")
    subprocess.run(["python3", "-m", "lincs", "generate", "classification-domain", "4", "3"], check=True)
    print()
    print("Use as a Python package")
    subprocess.run(
        ["python3"],
        check=True,
        # @todo Reduce this example, turn its tests into unit tests
        input=textwrap.dedent("""
            import io
            import sys
            import lincs

            criterion = lincs.Criterion('Physic grade', lincs.ValueType.real, lincs.CategoryCorrelation.growing)
            print(criterion.name, criterion.value_type, criterion.category_correlation)
            criterion.name = 'Physics grade'
            criterion.value_type = lincs.ValueType.real
            criterion.category_correlation = lincs.CategoryCorrelation.growing

            domain = lincs.Domain(
                [
                    criterion,
                    lincs.Criterion('Literature grade', lincs.ValueType.real, lincs.CategoryCorrelation.growing),
                ],
                (
                    lincs.Category('Bad'),
                    lincs.Category('Good'),
                ),
            )

            criterion = domain.criteria[0]
            print(criterion.name, criterion.value_type, criterion.category_correlation)
            print(domain.categories[0].name)
            domain.categories[0].name = "Terrible"
            domain.categories[1].name = "Ok, I guess"

            buf = io.StringIO()
            domain.dump(buf)
            print(buf.getvalue().rstrip())

            model = lincs.Model(domain, [lincs.Boundary([10.,10.], lincs.SufficientCoalitions(lincs.SufficientCoalitionsKind.weights, [0.4, 0.7]))])
            model.dump(sys.stdout)

            alternatives = lincs.AlternativesSet(domain, [lincs.Alternative('Alice', [11., 12.], 'Good'), lincs.Alternative('Bob', [9., 11.], 'Bad')])
            alternatives.dump(sys.stdout)
        """),
        universal_newlines=True,
    )
    print()
    print("Use as a C++ library")
    subprocess.run(
        [
            "g++",
            "-x", "c++", "-",
            "-I/home/user/.local/lib/python3.10/site-packages/lincs/liblincs",
            "-L/home/user/.local/lib/python3.10/site-packages", "-llincs.cpython-310-x86_64-linux-gnu",
        ],
        check=True,
        # @todo Reduce this example, turn its tests into unit tests
        input=textwrap.dedent("""
            #include <lincs.hpp>

            #include <iostream>
            #include <sstream>

            int main() {
                lincs::Domain domain{
                    {
                        {"Literature grade", lincs::Domain::Criterion::ValueType::real, lincs::Domain::Criterion::CategoryCorrelation::growing},
                        {"Physics grade", lincs::Domain::Criterion::ValueType::real, lincs::Domain::Criterion::CategoryCorrelation::growing},
                    },
                    {
                        {"Fail"},
                        {"Pass"},
                    }
                };

                domain.dump(std::cout);

                lincs::Model model{&domain, {{{10.f, 10.f}, {lincs::Model::SufficientCoalitions::Kind::weights, {0.4f, 0.7f}}}}};
                {
                    std::ostringstream oss;
                    model.dump(oss);
                    std::cout << oss.str() << std::endl;
                    std::istringstream iss(oss.str());
                    lincs::Model model2 = lincs::Model::load(&domain, iss);

                    model2.dump(std::cout);
                }

                lincs::AlternativesSet alternatives{&domain, {{"Alice", {11.f, 12.f}, "Pass"}, {"Bob", {9.f, 11.f}, "Fail"}}};
                {
                    std::ostringstream oss;
                    alternatives.dump(oss);
                    std::cout << oss.str();
                    std::istringstream iss(oss.str());
                    lincs::AlternativesSet alternatives2 = lincs::AlternativesSet::load(&domain, iss);

                    alternatives2.dump(std::cout);
                }
            }
        """),
        universal_newlines=True,
    )
    subprocess.run(["./a.out"], check=True, env={"LD_LIBRARY_PATH": "/home/user/.local/lib/python3.10/site-packages"})

    with open("/wd/lincs-help-all.txt", "w") as f:
        subprocess.run(["lincs", "help-all"], stdout=f, check=True)


if __name__ == "__main__":
    main()
