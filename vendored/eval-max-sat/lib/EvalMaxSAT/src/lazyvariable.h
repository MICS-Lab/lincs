#ifndef IMPLICATIONGRAPH_H
#define IMPLICATIONGRAPH_H

#include <optional>
#include <vector>
#include <memory>
#include <cassert>

#include "virtualcard.h"
/* Removed for lincs
#include "MaLib/coutUtil.h"
*/  // Removed for lincs

class VirtualSAT;

class LazyVariable {

    VirtualSAT *solver;

    std::optional<int> var = {};

   // For a node without 'var'. Lists clauses ; one of them should be satisfied for this variable to be "on" for the unary number.
    std::vector< std::vector< std::shared_ptr<LazyVariable> > > impliquants;

    LazyVariable( VirtualSAT *solver )
        : solver(solver) {
    }


public:

    // Create a new LazyVariable instance with var and return it - TODO : can't this be a constructor instead ?
    static std::shared_ptr<LazyVariable> encapsulate(int variable) {
        assert(variable != 0);
        auto result = std::shared_ptr<LazyVariable>(new LazyVariable(nullptr));
        result->var = variable;
        return result;
    }

    // Create a new LazyVariable instance with a SatSolver instead - TODO : could be a constructor too
    static std::shared_ptr<LazyVariable> newVar(VirtualSAT *solver) {
        return std::shared_ptr<LazyVariable>(new LazyVariable(solver));
    }

    // (\wedge_{v \in lazyVars} v) => this
    void addImpliquant(const std::vector< std::shared_ptr<LazyVariable> > &vars) {
        assert(vars.size() > 0);
        impliquants.push_back( vars );
    }

    int get();
};










#endif // IMPLICATIONGRAPH_H
