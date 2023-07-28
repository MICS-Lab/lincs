#include "lazyvariable.h"

#include "MaLib/coutUtil.h"
#include "virtualsat.h"

// Gets value of current LazyVariable
int LazyVariable::get() {

    if(!var) {
        assert(impliquants.size() > 0);

        // End case in which you have a parent node with only one leaf. Recursive call on this leaf to get its var.
        if(impliquants.size() == 1) {
            assert(impliquants[0].size() > 0);
            if(impliquants[0].size() == 1 ) {
                var = impliquants[0][0]->get();
                return *var;
            }
        }

        // To revisit - the name of the method is misleading as we're actually adding a hard var
        // var = solver->newSoftVar(false, 0);      // TODO : question : pourquoi newSoftVar ??
        var = solver->newVar(false);

        /*
         * Add the cardinality constraints to the SatSolver in a recursive manner.
         * Example follows. If we have the vars O1 ... 05 at root, then all the combinations of lits will be added as
         * a clause, with a soft var at the end to make it optional ("soft") :
         * { l1, <softV> }, ..., { l1, l2, l3, <softV> }, ... , { l1, l2, l3, l4, l5, <soft> }
         */
        for(auto &implique: impliquants) {
            std::vector<int> clause;
            for(auto &lazyVar: implique) {
                int newVar = lazyVar->get();
                assert( newVar != 0);
                clause.push_back(-newVar);
            }
            assert(clause.size() > 0);

            clause.push_back(*var);

            solver -> addClause( clause );
        }
    }

    return *var;
}

