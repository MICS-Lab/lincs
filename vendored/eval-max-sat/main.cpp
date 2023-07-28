#include <iostream>
#include <cassert>
#include <csignal>
#include <zlib.h>

#include "EvalMaxSAT.h"
#include "lib/CLI11.hpp"

// Pour les tests
#include "unweighted_data.h"
#include "weighted_data.h"

#include "config.h"

using namespace MaLib;

EvalMaxSAT* monMaxSat = nullptr;

std::string cur_file;
void signalHandler( int signum ) {
    std::cout << "c Interrupt signal (" << signum << ") received, curFile = "<<cur_file<< std::endl;
    std::cout << "c o >=" << monMaxSat->getCost() << std::endl;
    std::cout << "s UNKNOWN" << std::endl;

   delete monMaxSat;

   exit(signum);
}


int main(int argc, char *argv[])
{
    Chrono chrono("c Total time");
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    /////// PARSE ARG //////////////////////
    CLI::App app{"EvalMaxSAT Solver"};

    std::string file;
    app.add_option("file", file, "File with the formula to be solved (wcnf format)")->check(CLI::ExistingFile)->required();

    unsigned int paralleleThread=0;
    app.add_option("-p", paralleleThread, toString("Number of minimization threads (default = ",paralleleThread,")"));

    unsigned int timeOutFastMinimize=60;
    app.add_option("--timeout_fast", timeOutFastMinimize, toString("Timeout in second for fast minimize (default = ",timeOutFastMinimize,")"));

    unsigned int coefMinimizeTime=2;
    app.add_option("--coef_minimize", coefMinimizeTime, toString("Multiplying coefficient of the time spent to minimize cores (default = ",coefMinimizeTime,")"));

    bool oldOutputFormat = false;
    app.add_flag("--old", oldOutputFormat, "Use old output format.");

    bool bench = false;
    app.add_flag("--bench", bench, "Print result in one line");

    CLI11_PARSE(app, argc, argv);
    ////////////////////////////////////////
    cur_file = file;

    auto monMaxSat = new EvalMaxSAT(paralleleThread);
    monMaxSat->setTimeOutFast(timeOutFastMinimize);
    monMaxSat->setCoefMinimize(coefMinimizeTime);

    if(!monMaxSat->parse(file)) {
        return -1;
    }

    if(!monMaxSat->solve()) {
        std::cout << "s UNSATISFIABLE" << std::endl;
        return 0;
    }


    if(bench) {
        std::cout << file << "\t" << monMaxSat->getCost() << "\t" << chrono.tacSec() << std::endl;
        chrono.afficherQuandDetruit(false);
        C_solve.afficherQuandDetruit(false);
        C_fastMinimize.afficherQuandDetruit(false);
        C_fullMinimize.afficherQuandDetruit(false);
        C_extractAM.afficherQuandDetruit(false);
        C_harden.afficherQuandDetruit(false);
        C_extractAMAfterHarden.afficherQuandDetruit(false);
    } else if(oldOutputFormat) {
        ////// PRINT SOLUTION OLD FORMAT //////////////////
        std::cout << "s OPTIMUM FOUND" << std::endl;
        std::cout << "o " << monMaxSat->getCost() << std::endl;
        std::cout << "v";
        for(unsigned int i=1; i<=monMaxSat->nInputVars; i++) {
            if(monMaxSat->getValue(i))
                std::cout << " " << i;
            else
                std::cout << " -" << i;
        }
        std::cout << std::endl;
        ///////////////////////////////////////
    } else {
        ////// PRINT SOLUTION NEW FORMAT //////////////////
        std::cout << "s OPTIMUM FOUND" << std::endl;
        std::cout << "o " << monMaxSat->getCost() << std::endl;
        std::cout << "v ";
        for(unsigned int i=1; i<=monMaxSat->nInputVars; i++) {
            std::cout << monMaxSat->getValue(i);
        }
        std::cout << std::endl;
        ///////////////////////////////////////
    }


    delete monMaxSat;
    return 0;
}



