#ifndef EVALMAXSAT_SLK178903R_H
#define EVALMAXSAT_SLK178903R_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <map>

#include "MaLib/communicationlist.h"
#include "MaLib/Chrono.h"
/* Removed for lincs
#include "MaLib/coutUtil.h"
*/  // Removed for lincs
#include "virtualmaxsat.h"
#include "virtualsat.h"
#include "cadicalinterface.h"
#include "mcqd.h"
/* Removed for lincs
#include "MaLib/coutUtil.h"
*/  // Removed for lincs

using namespace MaLib;

/* Removed for lincs
MaLib::Chrono C_solve("c Cumulative time spent solving SAT formulas");
MaLib::Chrono C_fastMinimize("c Cumulative time spent for fastMinimize");
MaLib::Chrono C_fullMinimize("c Cumulative time spent for fullMinimize");
MaLib::Chrono C_extractAM("c Cumulative time spent for extractAM");
MaLib::Chrono C_harden("c Cumulative time spent for harden");
MaLib::Chrono C_extractAMAfterHarden("c Cumulative time spent for extractAM afterHarden");
*/  // Removed for lincs

template<class B>
static void readClause(B& in, std::vector<int>& lits) {
    int parsed_lit;
    lits.clear();
    for (;;){
        parsed_lit = parseInt(in);
        if (parsed_lit == 0) break;
        lits.push_back( parsed_lit );
    }
}

/* Removed for lincs
inline t_weight calculateCost(const std::string & file, const std::vector<bool> &result) {
    t_weight cost = 0;
    auto in_ = gzopen(file.c_str(), "rb");
    t_weight weightForHardClause = -1;

    StreamBuffer in(in_);

    std::vector<int> lits;
    for(;;) {
        skipWhitespace(in);

        if(*in == EOF)
            break;

        else if(*in == 'c') {
            skipLine(in);
        } else if(*in == 'p') { // Old format
          ++in;
          if(*in != ' ') {
              std::cerr << "o PARSE ERROR! Unexpected char: " << static_cast<char>(*in) << std::endl;
              return false;
          }
          skipWhitespace(in);

          if(eagerMatch(in, "wcnf")) {
              parseInt(in); // # Var
              parseInt(in); // # Clauses
              weightForHardClause = parseWeight(in);
          } else {
              std::cerr << "o PARSE ERROR! Unexpected char: " << static_cast<char>(*in) << std::endl;
              return false;
          }
      }
        else {
            t_weight weight = parseWeight(in);
            readClause(in, lits);
            if(weight == weightForHardClause) {
                bool sat=false;
                for(auto l: lits) {
                    if(abs(l) >= result.size()) {
                        std::cerr << "calculateCost: Parsing error." << std::endl;
                        return -1;
                    }
                    if ( (l>0) == (result[abs(l)]) ) {
                        sat = true;
                        break;
                    }
                }
                if(!sat) {
                    std::cerr << "calculateCost: NON SAT !" << std::endl;
                    return -1;
                }
            } else {
                bool sat=false;
                for(auto l: lits) {
                    if(abs(l) >= result.size()) {
                        std::cerr << "calculateCost: Parsing error." << std::endl;
                        return -1;
                    }

                    if ( (l>0) == (result[abs(l)]) ) {
                        sat = true;
                        break;
                    }
                }
                if(!sat) {
                    cost += weight;
                }
            }
        }
    }

    gzclose(in_);
    return cost;
}
*/  // Removed for lincs

class EvalMaxSAT : public VirtualMAXSAT {
    unsigned int nbMinimizeThread;

    VirtualSAT *solver = nullptr;
    //std::vector<VirtualSAT*> solverForMinimize;

    //int nVars = 0;
    int nVarsInSolver;

    std::vector<t_weight> _weight; // Weight of var at index, 0 if hard
    std::vector<bool> model; // Sign of var at index
    std::vector< std::tuple<int, unsigned int> > mapAssum2cardAndK; // Soft var as index to get <the index of CardIncremental_Lazy object in save_card, the k associed to the var>
    std::vector< std::tuple<std::shared_ptr<VirtualCard>, int, t_weight> > save_card; // Contains CardIncremental_Lazy objects, aka card. constraints

    std::map<t_weight, std::set<int>> mapWeight2Assum; // Used for the weighted case

    MaLib::Chrono chronoLastSolve;
    MaLib::Chrono mainChronoForSolve;

    std::atomic<t_weight> cost = 0;
    unsigned int _timeOutFastMinimize=60; // TODO: Magic number
    unsigned int _coefMinimizeTime = 2.0; // TODO: Magic number
    double _percentageMinForStratify = 0; // TODO: Magic number
    double _speedIncreasePercentageMinForStratify = 0.03; // TODO: Magic number : add 0.03 each minute

    ///// For communication between threads
    MaLib::CommunicationList< std::tuple<std::list<int>, long> > CL_ConflictToMinimize;
    MaLib::CommunicationList< int > CL_LitToUnrelax; // Variables to remove from the assumptions and put back in core
    MaLib::CommunicationList< int > CL_LitToRelax; // Variables to try to relax the cardinality constraint with which they're related
    MaLib::CommunicationList< std::tuple<std::vector<int>, bool, t_weight> > CL_CardToAdd; // Cores for which to add cardinality constraints
    std::atomic<unsigned int> maximumNumberOfActiveMinimizingThread;
    /////



    struct CompLit {
      bool operator() (const int& a, const int& b) const {
          if(abs(a) < abs(b))
              return true;

          if(abs(a) == abs(b))
              return (a>0) && (b<0);

          return false;
      }
    };

    std::set<int> _assumption;

    //////////////////////////////
    ////// For extractAM ////////
    ////////////////////////////
    void extractAM() {
        adapt_am1_exact();
        adapt_am1_FastHeuristicV7();
    }

   void reduceCliqueV2(std::list<int> & clique) {
       if(isWeighted()) {
           clique.sort([&](int litA, int litB){
               return _weight[ abs(litA) ] < _weight[ abs(litB) ];
           });
       }

       for(auto posImpliquant = clique.begin() ; posImpliquant != clique.end() ; ++posImpliquant) {
           auto posImpliquant2 = posImpliquant;
           for(++posImpliquant2 ; posImpliquant2 != clique.end() ; ) {
               if(solver->solveLimited(std::vector<int>({-(*posImpliquant), -(*posImpliquant2)}), 10000) != 0) { // solve != UNSAT
                   posImpliquant2 = clique.erase(posImpliquant2);
               } else {
                   ++posImpliquant2;
               }
           }
       }
   }

   bool adapt_am1_FastHeuristicV7() {
       /* Removed for lincs
       MonPrint("adapt_am1_FastHeuristic : (_weight.size() = ", _weight.size(), " )");
       */  // Removed for lincs

       Chrono chrono;
       std::vector<int> prop;
       unsigned int nbCliqueFound=0;

       // TODO : trier les var en fonction du nombre de prop.size()    apres solver->propagate({LIT}, prop)

       // Where nVarsInSolver represents the number of vars before the cardinality constraints. We don't want to
       // propagate soft vars representing cardinality constraints.     // TODO: pour quoi pas ?
       for(unsigned int VAR = 1; VAR<_weight.size(); VAR++) {    // TODO : Utiliser mapWeight2Assum pour eviter de parcourire tout _weight
           if(_weight[VAR] == 0)
               continue;

           assert(_weight[VAR] > 0);
           int LIT = model[VAR]?static_cast<int>(VAR):-static_cast<int>(VAR);
           prop.clear();
           if(solver->propagate({LIT}, prop)) {
               if(prop.size() == 0)
                   continue;

               std::list<int> clique;
               for(auto litProp: prop) {
                   if(isInAssum(-litProp)) {
                       clique.push_back(litProp);
                       assert(solver->solve(std::vector<int>({-litProp, LIT})) == false);
                   }
               }

               if(clique.size() == 0)
                   continue;

               reduceCliqueV2(clique); // retirer des elements pour que clique soit une clique

               clique.push_back(-LIT);

               if(clique.size() >= 2) {
                   nbCliqueFound++;

                   std::vector<int> clause;
                   for(auto lit: clique)
                       clause.push_back(-lit);

                   processAtMostOne(clause);
               }
           } else {
               addClause({-LIT});
           }
       }

       /* Removed for lincs
       MonPrint(nbCliqueFound, " cliques found in ", chrono.tac() / 1000, "ms.");
       */  // Removed for lincs
       return true;
   }

   bool adapt_am1_exact() {
       Chrono chrono;
       unsigned int nbCliqueFound=0;
       std::vector<int> assumption;

       for(auto & [w, lits]: mapWeight2Assum) {
           assert(w != 0);
           for(auto lit: lits) {
               assert( model[abs(lit)]== (lit>0) );
               assumption.push_back(lit);
           }
       }

       /* Removed for lincs
       MonPrint("Nombre d'assumption: ", assumption.size());
       */  // Removed for lincs

       if(assumption.size() > 30000) { // hyper paramétre
           /* Removed for lincs
           MonPrint("skip");
           */  // Removed for lincs
           return false;
       }

       /* Removed for lincs
       MonPrint("Create graph for searching clique...");
       */  // Removed for lincs
       unsigned int size = assumption.size();
       bool **conn = new bool*[size];
       for(unsigned int i=0; i<size; i++) {
           conn[i] = new bool[size];
           for(unsigned int x=0; x<size; x++)
               conn[i][x] = false;
       }

       /* Removed for lincs
       MonPrint("Create link in graph...");
       */  // Removed for lincs
       for(unsigned int i=0; i<size; ) {
           int lit1 = assumption[i];


           std::vector<int> prop;
           // If literal in assumptions has a value that is resolvable, get array of all the other literals that must have
           // a certain value in consequence, then link said literal to the opposite value of these other literals in graph

           if(solver->propagate({lit1}, prop)) {
               for(int lit2: prop) {
                   for(unsigned int j=0; j<size; j++) {
                       if(j==i)
                           continue;
                       if(assumption[j] == -lit2) {
                           conn[i][j] = true;
                           conn[j][i] = true;
                       }
                   }
               }
               i++;
           } else { // No solution - Remove literal from the assumptions and add its opposite as a clause
               addClause({-lit1});

               assumption[i] = assumption.back();
               assumption.pop_back();

               for(unsigned int x=0; x<size; x++) {
                   conn[i][x] = false;
                   conn[x][i] = false;
               }

               size--;
           }
       }


       if(size == 0) {
           for(unsigned int i=0; i<size; i++) {
               delete [] conn[i];
           }
           delete [] conn;
           return true;
       }

       std::vector<bool> active(size, true);
       for(;;) {
           int *qmax;
           int qsize=0;
           Maxclique md(conn, size, 0.025);
           md.mcqdyn(qmax, qsize, 100000);

           if(qsize <= 2) { // Hyperparametre: Taille minimal a laquelle arreter la methode exact
               for(unsigned int i=0; i<size; i++) {
                   delete [] conn[i];
               }
               delete [] conn;
               delete [] qmax;

               /* Removed for lincs
               MonPrint(nbCliqueFound, " cliques found in ", (chrono.tac() / 1000), "ms.");
               */  // Removed for lincs
               return true;
           }
           nbCliqueFound++;

           {
               //int newI=qmax[0];
               std::vector<int> clause;

               for (unsigned int i = 0; i < qsize; i++) {
                   int lit = assumption[qmax[i]];
                   active[qmax[i]] = false;
                   clause.push_back(lit);

                   for(unsigned int x=0; x<size; x++) {
                       conn[qmax[i]][x] = false;
                       conn[x][qmax[i]] = false;
                   }
               }
               auto newAssum = processAtMostOne(clause);
               assert(qsize >= newAssum.size());

               for(unsigned int j=0; j<newAssum.size() ; j++) {
                   assumption[ qmax[j] ] = newAssum[j];
                   active[ qmax[j] ] = true;

                   std::vector<int> prop;
                   if(solver->propagate({newAssum[j]}, prop)) {
                       for(int lit2: prop) {
                           for(unsigned int k=0; k<size; k++) {
                               if(active[k]) {
                                   if(assumption[k] == -lit2) {
                                       conn[qmax[j]][k] = true;
                                       conn[k][qmax[j]] = true;
                                   }
                               }
                           }
                       }
                    } else {
                       assert(solver->solve(std::vector<int>({newAssum[j]})) == false);
                       addClause({-newAssum[j]});
                    }
               }
           }

           delete [] qmax;
       }

       assert(false);
   }

   // Harden soft vars in passed clique to then unrelax them via a new cardinality constraint
   std::vector<int> processAtMostOne(std::vector<int> clause) {
       // Works also in the weighted case
       std::vector<int> newAssum;

       while(clause.size() > 1) {

           assert([&](){
               for(unsigned int i=0; i<clause.size(); i++) {
                   for(unsigned int j=i+1; j<clause.size(); j++) {
                       assert(solver->solve(std::vector<int>({clause[i], clause[j]})) == 0 );
                   }
               }
               return true;
           }());

           auto saveClause = clause;
           t_weight w = _weight[ abs(clause[0]) ];

           for(unsigned int i=1; i<clause.size(); i++) {
               if( w > _weight[ abs(clause[i]) ] ) {
                   w = _weight[ abs(clause[i]) ];
               }
           }
           assert(w > 0);

           for(unsigned int i=0; i<clause.size(); ) {
               assert( model[ abs(clause[i]) ] == (clause[i]>0) );

               assert( mapWeight2Assum[ _weight[ abs(clause[i]) ] ].count( clause[i] ) );
               mapWeight2Assum[ _weight[ abs(clause[i]) ] ].erase( clause[i] );
               _weight[ abs(clause[i]) ] -= w;

               if( _weight[ abs(clause[i]) ] == 0 ) {
                   _assumption.erase( clause[i] );
                   relax( clause[i] );
                   clause[i] = clause.back();
                   clause.pop_back();
               } else {
                   mapWeight2Assum[ _weight[ abs(clause[i]) ] ].insert( clause[i] );
                   i++;
               }
           }
           /* Removed for lincs
           MonPrint("AM1: cost = ", cost, " + ", w * (t_weight)(saveClause.size()-1));
           */  // Removed for lincs
           cost += w * (t_weight)(saveClause.size()-1);

           assert(saveClause.size() > 1);
           newAssum.push_back( addWeightedClause(saveClause, w) );
           assert( _weight[ abs(newAssum.back()) ] > 0 );
           assert( model[ abs(newAssum.back()) ]  == (newAssum.back() > 0) );
       }

       if( clause.size() ) {
           newAssum.push_back(clause[0]);
       }
       return newAssum;
   }





public:
    EvalMaxSAT(unsigned int nbMinimizeThread=0, VirtualSAT *solver =
            new CadicalInterface()
    ) : nbMinimizeThread(nbMinimizeThread), solver(solver)
    {
        //for(unsigned int i=0; i<nbMinimizeThread; i++) {
        //    solverForMinimize.push_back(new CadicalInterface());
        //}

        _weight.push_back(0);                   //
        model.push_back(false);                 // Fake lit with id=0
        mapAssum2cardAndK.push_back({-1, 0});   //


        /* Removed for lincs
        C_solve.pause(true);
        C_fastMinimize.pause(true);
        C_fullMinimize.pause(true);
        C_extractAM.pause(true);
        C_harden.pause(true);
        C_extractAMAfterHarden.pause(true);
        */  // Removed for lincs
    }

    virtual ~EvalMaxSAT();

   void addClause( const std::vector<int> &clause) override {
       if(clause.size() == 1) {
           if(_weight[abs(clause[0])] != 0) {

               if( (clause[0]>0) == model[abs(clause[0])] ) {
                   assert( mapWeight2Assum[ _weight[abs(clause[0])] ].count( clause[0] ) );
                   mapWeight2Assum[ _weight[abs(clause[0])] ].erase( clause[0] );
                   _weight[abs(clause[0])] = 0;
                   _assumption.erase(clause[0]);
               } else {
                   assert( mapWeight2Assum[ _weight[abs(clause[0])] ].count( -clause[0] ) );
                   mapWeight2Assum[ _weight[abs(clause[0])] ].erase( -clause[0] );
                   cost += _weight[abs(clause[0])];
                   _weight[abs(clause[0])] = 0;
                   _assumption.erase(-clause[0]);
                   relax(-clause[0]);
               }
           }
       }

       solver->addClause( clause );
   }

    virtual void simplify() {
        assert(!"TODO");
    }

    virtual unsigned int nVars() override {
        return solver->nVars();
    }

    virtual bool solve(const std::vector<int> &assumption) {
        assert(!"TODO");
        return false;
    }

    virtual int solveLimited(const std::vector<int> &assumption, int confBudget) {
        assert(!"TODO");
        return 0;
    }

    virtual std::vector<int> getConflict() {
        assert(!"TODO");
        return {};
    }


    bool isInAssum(int lit) {
        unsigned int var = static_cast<unsigned int>(abs(lit));
        if( _weight[var] > 0 ) {
            if( model[var] == (lit>0) )
                return true;
        }
        return false;
    }

    private:

    void minimize(VirtualSAT* S, std::list<int> & conflict, long refTime, bool doFastMinimize) {
        auto saveconflict = conflict;

        std::vector<int> uselessLit;
        std::vector<int> L;
        bool completed=false;
        t_weight minWeight = std::numeric_limits<t_weight>::max();
        if( (!doFastMinimize) ) {
            if( (mainChronoForSolve.tacSec() < (3600/2)) ) {
                std::set<int> conflictMin(conflict.begin(), conflict.end());
                completed = fullMinimize(S, conflictMin, uselessLit, _coefMinimizeTime*refTime);
                conflict = std::list<int>(conflictMin.begin(), conflictMin.end());
            } else {
                completed = fullMinimizeOneIT(S, conflict, uselessLit);
            }
        } else {
            /* Removed for lincs
            MonPrint("FullMinimize: skip");
            */  // Removed for lincs
        }

        for(auto lit: conflict) {
            L.push_back(-lit);
            if(minWeight > _weight[abs(lit)]) {
                minWeight = _weight[abs(lit)];
            }
        }
        assert(minWeight > 0);
        for(auto lit: conflict) {
            assert( mapWeight2Assum[ _weight[abs(lit)] ].count( lit ));
            mapWeight2Assum[ _weight[abs(lit)] ].erase( lit );
            _weight[abs(lit)] -= minWeight;
            if( _weight[abs(lit)] != 0) {
                uselessLit.push_back( lit );
                mapWeight2Assum[ _weight[abs(lit)] ].insert( lit );
            } else {
                if(std::get<0>(mapAssum2cardAndK[abs(lit)]) != -1) {
                    CL_LitToRelax.push(lit);
                }
            }
        }

        /* Removed for lincs
        MonPrint("\t\t\tMain Thread: cost = ", cost, " + ", minWeight);
        */  // Removed for lincs
        cost += minWeight;

        CL_LitToUnrelax.pushAll(uselessLit);
        if(L.size() > 1) {
            CL_CardToAdd.push({L, !completed, minWeight});
        }

        /* Removed for lincs
        MonPrint("size conflict after Minimize: ", conflict.size());
        */  // Removed for lincs
    }

    void threadMinimize(unsigned int num, VirtualSAT* solverForMinimize, bool fastMinimize) {
        for(;;) {
            auto element = CL_ConflictToMinimize.pop();
            /* Removed for lincs
            MonPrint("threadMinimize[",num,"]: Run...");
            */  // Removed for lincs

            if(!element) {
                break;
            }

            minimize(solverForMinimize, std::get<0>(*element), std::get<1>(*element), fastMinimize);
        }

        delete solverForMinimize;
    }

    void apply_CL_CardToAdd() {
        while(CL_CardToAdd.size()) {
            // Each "set" in CL_CardToAdd contains the literals of a core
            auto element = CL_CardToAdd.pop();
            assert(element);

            std::shared_ptr<VirtualCard> card = std::make_shared<CardIncremental_Lazy>(this, std::get<0>(*element), 1);
            //std::shared_ptr<VirtualCard> card = std::make_shared<Card_Lazy_OE>(this, std::get<0>(*element));


            // save_card contains our cardinality constraints
            save_card.push_back( {card, 1, std::get<2>(*element)} );

            int k = 1;

            int newAssumForCard = card->atMost(k); // Gets the soft variable corresponding to the cardinality constraint with RHS = 1

            assert(newAssumForCard != 0);

            // TODO: Exhaust semble n'avoir pas d'impacte sur les performences ?
/*
            MonPrint("solveLimited for Exhaust...");
            if(std::get<1>(*element)) { // if clause hasn't been fully minimized
                // Relax (inc) while the cardinality constraint cannot be satisfied with no other assumptions ; aka exhaust
                while(solver->solveLimited(std::vector<int>({newAssumForCard}), 10000) == 0) { // solve == UNSAT
                    k++;
                    std::cout << "Exhaust !!!!!" << std::endl;
                    MonPrint("cost = ", cost, " + ", std::get<2>(*element));
                    cost += std::get<2>(*element);
                    newAssumForCard = card->atMost(k);

                    if(newAssumForCard==0) {
                        break;
                    }
                }
                std::get<1>(save_card.back()) = k; // Update the rhs of the cardinality in the vector with its new value


            }
            MonPrint("Exhaust fini!");
            */


            if(newAssumForCard != 0) {
                assert( _weight[abs(newAssumForCard)] == 0 );
                _weight[abs(newAssumForCard)] = std::get<2>(*element);
                mapWeight2Assum[ _weight[abs(newAssumForCard)] ].insert( newAssumForCard );
                _assumption.insert( newAssumForCard );
                // Put cardinality constraint in mapAssum2cardAndK associated to softVar as index in mapAssum2cardAndK
                mapAssum2cardAndK[ abs(newAssumForCard) ] = {save_card.size()-1, k};
            }
        }
    }

    void apply_CL_LitToRelax() {
        while(CL_LitToRelax.size()) {
            int lit = CL_LitToRelax.pop().value_or(0);
            assert(lit != 0);
            relax(lit);
        }
    }

    // If a soft variable is not soft anymore, we have to check if this variable is a cardinality, in which case, we have to relax the cardinality.
    void relax(int lit) {
        assert(lit != 0);
        unsigned int var = abs(lit);
        assert( _weight[var] == 0 );
        _weight[var] = 0;
        if(std::get<0>(mapAssum2cardAndK[var]) != -1) { // If there is a cardinality constraint associated to this soft var
            int idCard = std::get<0>(mapAssum2cardAndK[var]); // Get index in save_card
            assert(idCard >= 0);

            // Note : No need to increment the cost here, as this cardinality constraint will be added inside another
            // cardinality constraint, and its non-satisfiability within it would implicate a cost increment anyway...

            std::get<1>(save_card[idCard])++; // Increase RHS

            // Get soft var associated with cardinality constraint with increased RHS
            int forCard = (std::get<0>(save_card[idCard])->atMost(std::get<1>(save_card[idCard])));

            if(forCard != 0) {
                _assumption.insert( forCard );
                assert( _weight[abs(forCard)] == 0 );
                _weight[abs(forCard)] = std::get<2>(save_card[idCard]);
                mapWeight2Assum[_weight[abs(forCard)]].insert( forCard );
                mapAssum2cardAndK[ abs(forCard) ] = {idCard, std::get<1>(save_card[idCard])};
            }
        }
    }


public:

    bool solve() override {
        std::mt19937 rng((std::random_device()()));

        mainChronoForSolve.tic();
        unsigned int nbSecondSolveMin = 20;      // TODO: Magic number
        unsigned int timeOutForSecondSolve = 60; // TODO: Magic number

        // Reinit CL
        CL_ConflictToMinimize.clear();
        CL_LitToUnrelax.clear();
        CL_LitToRelax.clear();
        CL_CardToAdd.clear();

        nVarsInSolver = nVars(); // Freeze nVarsInSolver in time

        /* Removed for lincs
        MonPrint("\t\t\tMain Thread: extractAM...");
        C_extractAM.pause(false);
        */  // Removed for lincs
        extractAM();
        /* Removed for lincs
        C_extractAM.pause(true);
        */  // Removed for lincs

        t_weight minWeightToConsider = chooseNextMinWeight();
        initializeAssumptions(minWeightToConsider);

        std::vector<std::thread> vMinimizeThread;
        vMinimizeThread.reserve(nbMinimizeThread);

         for(;;) {
            assert(CL_ConflictToMinimize.size()==0);
            assert(CL_LitToUnrelax.size()==0);
            assert(CL_LitToRelax.size()==0);
            assert(CL_CardToAdd.size()==0);
            maximumNumberOfActiveMinimizingThread = nbMinimizeThread;


            bool firstSolve = true;
            for(;;) {
                chronoLastSolve.tic();
                /* Removed for lincs
                MonPrint("\t\t\tMain Thread: Solve...");
                */  // Removed for lincs
                int resultSolve;
                /* Removed for lincs
                C_solve.pause(false);
                */  // Removed for lincs
                if(firstSolve) {
                    /* Removed for lincs
                    MonPrint("solve(",_assumption.size(),")...");
                    */  // Removed for lincs
                    resultSolve = solver->solve(_assumption); // 1 for SAT, 0 for UNSAT
                } else {
                    /* Removed for lincs
                    MonPrint("solveLimited(",_assumption.size(),")...");
                    */  // Removed for lincs
                    resultSolve = solver->solveLimited(_assumption, 10000); // 1 for SAT, 0 for UNSAT, -1 for UNKNOW
                }
                /* Removed for lincs
                C_solve.pause(true);
                */  // Removed for lincs

                if(resultSolve != 0) { // If last solve is not UNSAT
                    /* Removed for lincs
                    MonPrint("\t\t\tMain Thread: Solve() is not false!");
                    */  // Removed for lincs

                    if(firstSolve && minWeightToConsider==1) {
                        assert( resultSolve == 1 );
                        assert( CL_LitToUnrelax.size() == 0 );
                        assert( CL_CardToAdd.size() == 0 );
                        assert( CL_ConflictToMinimize.size() == 0 );
                        CL_ConflictToMinimize.close(); // Va impliquer la fin des threads minimize
                        for(auto &t: vMinimizeThread)
                            t.join();
                        return true;
                    }

/*
                    ///////////////
                    /// HARDEN ////
                    if(resultSolve == 1) { // If last solve is SAT
                        if(isWeighted()) {
                            if(harden()) {
                                C_extractAMAfterHarden.pause(false);
                                adapt_am1_FastHeuristicV7();
                                C_extractAMAfterHarden.pause(true);
                            }
                        }
                    } else {
                        // TODO: estimer cost sans qu'on est une solution
                    }
                    //////////////
*/

                    chronoLastSolve.pause(true);
                    /* Removed for lincs
                    MonPrint("\t\t\tMain Thread: CL_ConflictToMinimize.wait(nbMinimizeThread=",nbMinimizeThread,", true)...");
                    */  // Removed for lincs
                    CL_ConflictToMinimize.areWaiting(vMinimizeThread.size());
                    /* Removed for lincs
                    MonPrint("\t\t\tMain Thread: Fin boucle d'attente");
                    */  // Removed for lincs

                    ///////////////
                    /// HARDEN ////
                    if(resultSolve == 1) { // If last solve is SAT
                        if(isWeighted()) {
                            if(harden()) {
                                /* Removed for lincs
                                C_extractAMAfterHarden.pause(false);
                                */  // Removed for lincs
                                adapt_am1_FastHeuristicV7();
                                /* Removed for lincs
                                C_extractAMAfterHarden.pause(true);
                                */  // Removed for lincs
                            }
                        }
                    } else {
                        // TODO: estimer cost sans qu'on est une solution
                    }
                    //////////////


                    if(firstSolve) {
                        assert( resultSolve == 1 );
                        assert( CL_LitToUnrelax.size() == 0 );
                        assert( CL_CardToAdd.size() == 0 );
                        assert( CL_ConflictToMinimize.size() == 0 );

                        minWeightToConsider = chooseNextMinWeight(minWeightToConsider);
                        initializeAssumptions(minWeightToConsider);
                        break;
                    }


                    // If no variables are left to be unrelaxed, we are ready to consider the new cardinality constraints
                    if(CL_LitToUnrelax.size()==0) {
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: CL_LitToUnrelax.size()==0");

                        MonPrint("\t\t\tMain Thread: CL_LitToRelax.size() = ", CL_LitToRelax.size());
                        */  // Removed for lincs
                        apply_CL_LitToRelax();

                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: CL_CardToAdd.size() = ", CL_CardToAdd.size());
                        */  // Removed for lincs
                        apply_CL_CardToAdd();

                        break;
                    }

                    /* Removed for lincs
                    MonPrint("\t\t\tMain Thread: CL_LitToUnrelax.size()!=0");
                    */  // Removed for lincs
                } else { // Conflict found
                    /* Removed for lincs
                    MonPrint("\t\t\tMain Thread: Solve = false");
                    */  // Removed for lincs
                    chronoLastSolve.pause(true);

                    std::vector<int> bestUnminimizedConflict = solver->getConflict(_assumption);

                    // Special case in which the core is empty, meaning no solution can be found
                    if(bestUnminimizedConflict.empty()) {
                        cost = -1;
                        return false;
                    }

                    if(bestUnminimizedConflict.size() == 1) {
                        // TODO : si c'est une card, essayer de exhaust !!!
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: conflict size = 1");
                        MonPrint("\t\t\tMain Thread: cost = ", cost, " + ", _weight[ abs(bestUnminimizedConflict[0]) ]);
                        */  // Removed for lincs
                        cost += _weight[ abs(bestUnminimizedConflict[0]) ];

                        assert( mapWeight2Assum[_weight[abs(bestUnminimizedConflict[0])]].count( bestUnminimizedConflict[0] ) );
                        mapWeight2Assum[_weight[abs(bestUnminimizedConflict[0])]].erase( bestUnminimizedConflict[0] );
                        _weight[ abs(bestUnminimizedConflict[0]) ] = 0;
                        assert(_assumption.count(bestUnminimizedConflict[0]));
                        _assumption.erase(bestUnminimizedConflict[0]);
                        relax(bestUnminimizedConflict[0]);
                        //if(std::get<0>(mapAssum2cardAndK[abs(bestUnminimizedConflict[0])]) != -1) {
                        //  CL_LitToRelax.push(bestUnminimizedConflict[0]);
                        //  apply_CL_LitToRelax();
                        //}
                        continue;
                    }

                    MaLib::Chrono chronoForBreak;
                    unsigned int nbSecondSolve = 0;

                    if(_assumption.size() > 100000) {
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: Skip second solve...");
                        */  // Removed for lincs
                    } else {
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: Second solve...");
                        */  // Removed for lincs

                        // Shuffle assumptions in a loop to hopefully get a smaller core from the SatSolver
                        std::vector<int> forSolve(_assumption.begin(), _assumption.end());
                        while((nbSecondSolve < nbSecondSolveMin) || (chronoLastSolve.tac() >= chronoForBreak.tac())) {
                            if(bestUnminimizedConflict.size() == 1)
                                break;
                            nbSecondSolve++;
                            if(chronoForBreak.tacSec() > timeOutForSecondSolve)
                                break;
                            if(nbSecondSolve > 10000)
                                break;

                            std::shuffle(forSolve.begin(), forSolve.end(), rng);

                            bool res = solver->solve(forSolve);
                            assert(!res);

                            if( bestUnminimizedConflict.size() > solver->conflictSize() ) {
                                bestUnminimizedConflict = solver->getConflict(forSolve);
                            }
                        }
                    }

                    std::list<int> conflictMin;
                    for(auto lit: bestUnminimizedConflict)
                        conflictMin.push_back(lit);

                    bool doFullMinimize = true;
                    if((_assumption.size() < 100000) && (conflictMin.size() > 1)) {
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: fastMinimize(", conflictMin.size(), ")");
                        */  // Removed for lincs
                        // If the fastMinimize is timed out, don't execute the full one as it would be too long
                        doFullMinimize = fastMinimize(solver, conflictMin);
                    }

                    /* Removed for lincs
                    MonPrint("\t\t\tMain Thread: taille final du conflict = ", conflictMin.size());
                    */  // Removed for lincs

                    if(conflictMin.size() == 1) {
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: Optimal found, no need to fullMinimize");
                        */  // Removed for lincs
                        doFullMinimize = false;
                    }

                    if(doFullMinimize) {
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: call CL_ConflictToMinimize.push");
                        */  // Removed for lincs

                        // Remove problematic literals from the assumptions
                        for(auto lit: conflictMin) {
                            assert(_assumption.count(lit));
                            _assumption.erase(lit);
                        }
                        if(nbMinimizeThread == 0) {
                            minimize(solver, conflictMin, chronoLastSolve.tac(), _assumption.size() > 100000);
                        } else {

                            if( (vMinimizeThread.size() < nbMinimizeThread) && CL_ConflictToMinimize.getNumberWaiting() == 0) {
                                vMinimizeThread.emplace_back(&EvalMaxSAT::threadMinimize, this, vMinimizeThread.size(), solver->clone(), _assumption.size() > 100000);
                            }

                            CL_ConflictToMinimize.push({conflictMin, chronoLastSolve.tac()});
                        }

                        firstSolve = false;
                    } else {

                        t_weight minWeight = _weight[abs(*(conflictMin.begin()))];
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: new card");
                        */  // Removed for lincs
                        std::vector<int> L;
                        for(auto lit: conflictMin) {
                            L.push_back(-lit);
                            if(_weight[abs(lit)] < minWeight) {
                                minWeight = _weight[abs(lit)];
                            }
                        }
                        assert(minWeight > 0);
                        for(auto lit: conflictMin) {

                            assert( mapWeight2Assum[_weight[abs(lit)]].count( lit ) );
                            mapWeight2Assum[_weight[abs(lit)]].erase( lit );
                            _weight[abs(lit)] -= minWeight;
                            if(_weight[abs(lit)] == 0) {
                                if(std::get<0>(mapAssum2cardAndK[abs(lit)]) != -1) {
                                    CL_LitToRelax.push(lit);
                                }
                            } else {
                                mapWeight2Assum[_weight[abs(lit)]].insert( lit );
                            }
                        }

                        if(conflictMin.size() > 1) {
                            CL_CardToAdd.push({L, true, minWeight});
                        }
                        if(firstSolve) {
                            apply_CL_LitToRelax();      // TODO : On mesure une amélioration en effectuant apply maintenent ?
                            apply_CL_CardToAdd();       // TODO : On mesure une amélioration en effectuant apply maintenent ?
                        }

                        // Removal of literals that are no longer soft from the assumptions
                        for(auto lit: conflictMin) {
                            if(_weight[abs(lit)] == 0) {
                                assert(_assumption.count(lit));
                                _assumption.erase(lit);
                            }
                        }

                        assert(minWeight > 0);
                        /* Removed for lincs
                        MonPrint("\t\t\tMain Thread: cost = ", cost, " + ", minWeight);
                        */  // Removed for lincs
                        cost += minWeight;
                    }

                }

                while(CL_LitToUnrelax.size()) {
                    auto element = CL_LitToUnrelax.pop();
                    assert(element);
                    //assert(_weight[abs(*element)] > 0);
                    //if( _weight[abs(*element)] > minWeightToConsider) {
                    if(_weight[abs(*element)] > 0) {
                        _assumption.insert(*element);
                    }
                    //}
                }
            }

            CL_ConflictToMinimize.close(); // Va impliquer la fin des threads minimize
            for(auto &t: vMinimizeThread)
                t.join();
            CL_ConflictToMinimize.clear();
            vMinimizeThread.clear();


         }

    }

    void setTimeOutFast(unsigned int timeOutFastMinimize) {
        _timeOutFastMinimize = timeOutFastMinimize;
    }

    void setCoefMinimize(unsigned int coefMinimizeTime) {
        _coefMinimizeTime = coefMinimizeTime;
    }

    t_weight getCost() override {
        return cost;
    }

    bool getValue(unsigned int var) override {
        return solver->getValue(var);
    }


    virtual unsigned int newSoftVar(bool value, t_weight weight) override {
        if(weight > 1) {
            setIsWeightedVerif(); // TODO : remplacer par  mapWeight2Assum
        }
        _weight.push_back(weight);
        mapWeight2Assum[weight].insert( _weight.size()-1 );
        mapAssum2cardAndK.push_back({-1, 0});
        model.push_back(value);

        int var = solver->newVar();
        assert(var == _weight.size()-1);

        return var;
    }


    virtual int newVar(bool decisionVar=true) override {
        _weight.push_back(0);
        mapAssum2cardAndK.push_back({-1, 0});
        model.push_back(false);

        int var = solver->newVar(decisionVar);

        assert(var == _weight.size()-1);

        return var;
    }

    virtual bool isSoft(unsigned int var) override {
        return var < _weight.size() && _weight[var] > 0;
    }





    virtual void setVarSoft(unsigned int var, bool value, t_weight weight) override {
        while(var > nVars()) {
            newVar();
        }

        if( _weight[var] == 0 ) {
           _weight[var] = weight;
           mapWeight2Assum[_weight[var]].insert( (value?1:-1)*var );
           model[var] = value;      // "value" is the sign but represented as a bool
        } else {
            // In the case of weighted formula
            if(model[var] == value) {

                assert( mapWeight2Assum[_weight[var]].count( (value?1:-1)*var ) );
                mapWeight2Assum[_weight[var]].erase( (value?1:-1)*var );

                _weight[var] += weight;
            } else {

                assert( mapWeight2Assum[_weight[var]].count( -(value?1:-1)*var ) );
                mapWeight2Assum[_weight[var]].erase( -(value?1:-1)*var );

                if( _weight[var] > weight ) {
                    _weight[var] -= weight;
                    cost += weight;
                } else if( _weight[var] < weight ) {
                    cost += _weight[var];
                    _weight[var] = weight - _weight[var];
                    model[var] = !model[var];
                } else { assert( _weight[var] == weight );
                    cost += _weight[var];
                    _weight[var] = 0;
                    //nbSoft--;
                }
            }
            if(_weight[var] > 1) {
                setIsWeightedVerif(); // TODO : remplacer par  mapWeight2Assum
            }
            if(_weight[var] > 0) {
                mapWeight2Assum[_weight[var]].insert( (model[var]?1:-1)*var );
            } else {
                relax(var);
            }
        }

    }

    virtual unsigned int nSoftVar() override {
        unsigned int result = 0;
        for(auto w: _weight)
            if(w!=0)
                result++;
        return result;
    }

private:



    bool fullMinimize(VirtualSAT* solverForMinimize, std::set<int> &conflict, std::vector<int> &uselessLit, long timeRef) {
        std::mt19937 rng((std::random_device()()));

        /* Removed for lincs
        C_fullMinimize.pause(false);
        */  // Removed for lincs
        MaLib::Chrono chrono;
        bool minimum = true;

        int B = 1000;
        //int B = 10000;

        if(timeRef > 60000000) {
            timeRef = 60000000;  // Hyperparameter
        }

        std::vector<int> removable;
        /* Removed for lincs
        MonPrint("\t\t\t\t\tfullMinimize: Calculer Removable....");
        */  // Removed for lincs
        for(auto it = conflict.begin(); it != conflict.end(); ++it) {
            auto lit = *it;

            switch(solverForMinimize->solveLimited(conflict, B, lit)) {
            case -1: // UNKNOW
                minimum = false;
                [[fallthrough]];
            case 1: // SAT
                break;

            case 0: // UNSAT
                removable.push_back(lit);
                break;

            default:
                assert(false);
            }
        }
        /* Removed for lincs
        MonPrint("\t\t\t\t\tfullMinimize: removable = ", removable.size(), "/", conflict.size());
        */  // Removed for lincs

        if(removable.size() <= 1) {
            uselessLit = removable;
            for(auto lit: uselessLit) {
                conflict.erase(lit);
            }
            return minimum;
        }

        int maxLoop = 10000;
        if(removable.size() < 8) {
            maxLoop = 2*std::tgamma(removable.size()); // Gamma function is like a factorial but for natural numbers
        }


        if(isWeighted()) {
            std::sort(removable.begin(), removable.end(), [&](int litA, int litB){
                return _weight[ abs(litA) ] < _weight[ abs(litB) ];
            });
        }

        chrono.tic();
        // Same thing as above but with shuffles and a nested loop to hopefully find more useless lits
        for(int i=0; i<maxLoop; i++) {
            std::set<int> wConflict = conflict;
            std::vector<int> tmp_uselessLit;
            for(auto lit: removable) {
                switch(solverForMinimize->solveLimited(wConflict, B, lit)) {
                    case -1: // UNKNOW
                        minimum = false;
                        [[fallthrough]];
                    case 1: // SAT
                        break;

                    case 0: // UNSAT
                        wConflict.erase(lit);
                        tmp_uselessLit.push_back(lit);
                        break;

                    default:
                        assert(false);
                }
            }

            if(tmp_uselessLit.size() > uselessLit.size()) {
                /* Removed for lincs
                MonPrint("\t\t\t\t\tfullMinimize: newBest: ", tmp_uselessLit.size(), " removes.");
                */  // Removed for lincs
                uselessLit = tmp_uselessLit;
            }

            if(uselessLit.size() >= removable.size()-1) {
                /* Removed for lincs
                MonPrint("\t\t\t\t\tfullMinimize: Optimal trouvé.");
                */  // Removed for lincs
                break;
            }

            if((i>=2) // Au moins 3 loops
                    && (timeRef*(1+maximumNumberOfActiveMinimizingThread) <= chrono.tac())) {
                /* Removed for lincs
                MonPrint("\t\t\t\t\tfullMinimize: TimeOut after ", (i+1), " loops");
                */  // Removed for lincs
                break;
            }

            std::shuffle(removable.begin(), removable.end(), rng);
        }

        for(auto lit: uselessLit) {
            conflict.erase(lit);
        }

        /* Removed for lincs
        C_fullMinimize.pause(true);
        */  // Removed for lincs
        return minimum;
    }


    bool fullMinimizeOneIT(VirtualSAT* solverForMinimize, std::list<int> &conflict, std::vector<int> &uselessLit ) {
        /* Removed for lincs
        C_fullMinimize.pause(false);
        */  // Removed for lincs
        int B = 1000;
        //int B = 10000;

        if(isWeighted()) {
            conflict.sort( [&](int litA, int litB){
                return _weight[ abs(litA) ] < _weight[ abs(litB) ];
            });
        }

        for(auto it = conflict.begin(); it!=conflict.end(); ) {

            switch(solverForMinimize->solveLimited(conflict, B, *it)) {
            case -1:
                [[fallthrough]];
            case 1:
                ++it;
                break;

            case 0:
                uselessLit.push_back(*it);
                it = conflict.erase(it);
                break;

            default:
                assert(false);
            }
        }

        return true;
    }

    bool fastMinimize(VirtualSAT* solverForMinimize, std::list<int> &conflict) {
        /* Removed for lincs
        C_fastMinimize.pause(false);
        */  // Removed for lincs

        if(isWeighted()) {
            conflict.sort([&](int litA, int litB){
                return _weight[ abs(litA) ] < _weight[ abs(litB) ];
            });
        }

        int B = 1;
        Chrono chrono;
        for(auto it = conflict.begin(); it != conflict.end(); ++it) {

            if(chrono.tacSec() > _timeOutFastMinimize) {  // Hyperparameter
                /* Removed for lincs
                MonPrint("TIMEOUT fastMinimize!");
                C_fastMinimize.pause(true);
                */  // Removed for lincs
                return false;
            }

            auto lit = *it;
            it = conflict.erase(it);
            switch(solverForMinimize->solveLimited(conflict, B)) {
            case -1: // UNKNOW
                [[fallthrough]];
            case 1: // SAT
                it = conflict.insert(it, lit);
                break;

            case 0: // UNSAT
                break;

            default:
                assert(false);
            }
        }

        /* Removed for lincs
        C_fastMinimize.pause(true);
        */  // Removed for lincs
        return true;
    }

    virtual bool isWeighted() override {
        return mapWeight2Assum.size() > 1;
    }


    //////////////////////////////
    /// For weighted formula ////
    ////////////////////////////

    bool getValueImpliesByAssign(unsigned int var) {
        // TODO : ajouter un cache qui se vide apres chaque nouvel appel a solve ? (faire un benchmarking pour vérifier si ca vaut le coups)
        if(var < _mapSoft2clause.size()) {
            if(_mapSoft2clause[var].size()) {
                for(auto lit: _mapSoft2clause[var]) {
                    if( getValueImpliesByAssign( abs(lit) ) == (lit>0) ) {
                        return true;
                    }
                }
                return false;
            }
        }

        assert(var < mapAssum2cardAndK.size());
        auto [idCard, k] = mapAssum2cardAndK[var];
        if( idCard == -1 ) {
            return getValue(var);
        }

        unsigned int nb=0;
        for(auto lit: std::get<0>(save_card[ idCard ])->getClause()) {
            if( getValueImpliesByAssign( abs(lit) ) == (lit>0) ) {
                nb++;
            }
        }

        return !(nb <= k);
    }

    t_weight currentSolutionCost() {
        t_weight result = cost;

        // Consider lit not stratified
        for(auto & [w, lits]: mapWeight2Assum) {
            assert(w != 0);
            for(auto lit: lits) {
                auto var = abs(lit);
                assert(_weight[var] > 0);
                if( getValueImpliesByAssign(var) != model[var] ) {
                    assert( (_assumption.count(model[var]?var:-var) == 0) );
                    result += _weight[var];

                    if( std::get<0>(mapAssum2cardAndK[var]) != -1 ) {
                        auto [card, k, w] = save_card[ std::get<0>(mapAssum2cardAndK[var]) ];

                        unsigned int sum=0;
                        for(auto lit: card->getClause()) {
                            if( getValueImpliesByAssign( abs(lit) ) == (lit>0) ) {
                                sum++;
                            }
                        }
                        assert(sum > k); // car la card n'est pas satisfaite

                        result += ((t_weight)(sum-k-1)) * w;
                    }


                }
            }
        }




        // Consider the cards not yet created
        t_weight newCardCost = 0;
        for(const auto & [clause, exhaust, w]: CL_CardToAdd.getWithoutRemove_unsafe()) {
            unsigned int nb=0;
            for(auto lit: clause) {
                unsigned int var = abs(lit);
                assert( model[var] == (lit<0) );
                if( getValueImpliesByAssign( var ) == (lit>0) ) {
                    nb++;
                }
            }
            assert(nb >= 1);
            newCardCost += ((t_weight)nb - 1) * w;
        }
        result += newCardCost;

        // Consider lit to relax
        t_weight costFromRelax = 0;
        for(const int & lit: CL_LitToRelax.getWithoutRemove_unsafe()) {
            unsigned int var = abs(lit);

            auto [idCard, varK] = mapAssum2cardAndK[var];

            assert(idCard != -1);
            if(idCard != -1) { // If there is a cardinality constraint associated to this soft var
                assert(idCard >= 0);

                unsigned int k = std::get<1>(save_card[idCard]);
                assert(k == varK);
                t_weight w = std::get<2>(save_card[idCard]);
                unsigned int sum = 0;
                for(auto l: std::get<0>(save_card[idCard])->getClause()) {
                    if( getValueImpliesByAssign( abs(l) ) == (l>0) ) {
                        sum++;
                    }
                }

                if(sum > k) {
                    costFromRelax += ((t_weight)(sum-k-1)) * w;
                }
            }
        }
        result += costFromRelax;

        //std::cout << "c cost in ["<<cost<<", "<<result<<"]" << std::endl;

        return result;
    }

    // All soft variables whose cost is higher than the current solution can be considered as hard.
    unsigned int harden() {
        /* Removed for lincs
        C_harden.pause(false);
        */  // Removed for lincs

        auto costRemovedAssumLOCAL = currentSolutionCost();

        /* Removed for lincs
        assert([&](){
            C_harden.pause(true);
            std::vector<bool> assign;
            assign.push_back(0); // fake var_0
            for(unsigned int i=1; i<=nInputVars; i++) {
                assign.push_back(getValue(i));
            }
            auto costCalculated = calculateCost(savePourTest_file, assign);
            C_harden.pause(false);
            if(costRemovedAssumLOCAL != costCalculated) {
                std::cout << "savePourTest_file = " << savePourTest_file << std::endl;
                std::cout << "assign.size() = " << assign.size() << std::endl;
                std::cout << "costRemovedAssumLOCAL = " << costRemovedAssumLOCAL << std::endl;
                std::cout << "costCalculated = " << costCalculated << std::endl;
            }

            return costRemovedAssumLOCAL == costCalculated;
        }()); // POUR DEBUG : On vérifi que currentSolutionCost() estime corectement le cout
        */  // Removed for lincs
        costRemovedAssumLOCAL = costRemovedAssumLOCAL- cost;
        std::vector<int> unitClausesToAdd;
        for(auto it=mapWeight2Assum.rbegin(); it!=mapWeight2Assum.rend(); ++it) {
            if(it->first < costRemovedAssumLOCAL)
                break;
            for(auto lit: it->second) {
                auto var = abs(lit);
                assert( model[var] == (lit>0)  );
                if( getValueImpliesByAssign(var) == model[var]) {
                    unitClausesToAdd.push_back( lit );
                }
            }
        }

        for(auto lit: unitClausesToAdd) {
            addClause({lit});
        }
        /* Removed for lincs
        C_harden.pause(true);
        */  // Removed for lincs

        if(unitClausesToAdd.size()) {
            /* Removed for lincs
            MonPrint("\t\t\tMain Thread: ", unitClausesToAdd.size(), " harden !");
            */  // Removed for lincs
            assert(solver->solve(_assumption) == 1);
            //assert( harden() == 0 );
        }

        return unitClausesToAdd.size();
    }

    ///////////////////
    /// For stratify //
    ///////////////////

    t_weight chooseNextMinWeight(t_weight previousMinWeight = -1) {
        //return 1;   // Unactivate stratigy

        // clear empty mapWeight2Assum
        for(auto it = mapWeight2Assum.begin(); it != mapWeight2Assum.end(); ) {
            if(it->second.size() == 0) {
                it = mapWeight2Assum.erase(it);
            } else {
                ++it;
            }
        }

        unsigned int nbSoft = 0;
        for(auto &e: mapWeight2Assum) {
            nbSoft += e.first;
        }

        unsigned int nbNewConsidered = 0;
        unsigned int nbAlreadyConsidered = 0;
        unsigned int remainingLevel = mapWeight2Assum.size();
        for(auto it = mapWeight2Assum.rbegin() ; it != mapWeight2Assum.rend(); ++it, --remainingLevel) {

            if( it->first >= previousMinWeight ) {
                nbAlreadyConsidered += it->second.size();
                continue;
            }

            nbNewConsidered += it->second.size();

            /*
            /// Smallest
            t_weight result = it->first;
            ++it;
            if(it == mapWeight2Assum.rend())
                break;
            MonPrint("\t\t\tMain Thread: chooseNextMinWeight = ", result);
            return result;
            */
            if(nbSoft == nbAlreadyConsidered) { // Should not hapen
                /* Removed for lincs
                MonPrint("\t\t\tMain Thread: chooseNextMinWeight = 1");
                */  // Removed for lincs
                return 1;
            }

            // STRATÉGIE QUI AUGMENTE LE PAS DES STRATIFICATIONS AU FIL DU TEMPS
            if( ((double)nbNewConsidered / (double)(nbSoft - nbAlreadyConsidered)) >= _percentageMinForStratify + (mainChronoForSolve.tacSec()/60.0)*_speedIncreasePercentageMinForStratify ) {
                auto result = it->first;
                ++it;
                if(it == mapWeight2Assum.rend()) {
                    assert(remainingLevel == 1);
                    break;
                }
                /* Removed for lincs
                MonPrint("\t\t\tMain Thread: chooseNextMinWeight = ", result);
                */  // Removed for lincs
                return result;
            }

            /*
            if( ((double)nbNewConsidered / (double)(nbSoft)) >= 0.1) { // 10%   // TODO: trouver une stratégie
                auto result = it->first;
                ++it;
                if(it == mapWeight2Assum.rend()) {
                    assert(remainingLevel == 1);
                    break;
                }
                MonPrint("\t\t\tMain Thread: chooseNextMinWeight = ", result);
                return result;
            }
            */
        }

        /* Removed for lincs
        MonPrint("\t\t\tMain Thread: chooseNextMinWeight = 1");
        */  // Removed for lincs
        return 1;
    }

    void initializeAssumptions(t_weight minWeight) {
        _assumption.clear();

        for(auto it = mapWeight2Assum.rbegin(); it != mapWeight2Assum.rend(); ++it) {
            if(it->first < minWeight)
                break;
            for(auto lit: it->second) {
                _assumption.insert(lit);
            }
        }

    }

};



inline EvalMaxSAT::~EvalMaxSAT() {
    CL_ConflictToMinimize.close();
    CL_LitToUnrelax.close();
    CL_LitToRelax.close();
    CL_CardToAdd.close();

    delete solver;
}



#endif // EVALMAXSAT_SLK178903R_H

