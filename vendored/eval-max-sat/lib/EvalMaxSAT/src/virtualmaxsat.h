#ifndef VIRTUALMAXSAT_H
#define VIRTUALMAXSAT_H

#include <vector>
#include "virtualsat.h"
#include <set>
#include <fstream>
#include "cadical/cadical.hpp"
/* Removed for lincs
#include "cadical/file.hpp"
#include "ParseUtils.h"
*/  // Removed for lincs


typedef unsigned long long int t_weight;

class VirtualMAXSAT : public VirtualSAT {

    bool _isWeighted = false; // TODO : remplacer par  mapWeight2Assum
protected:
    //std::map<int, std::vector<int> > mapSoft2clause;    // which clause is related to which soft variable.
    std::vector< std::vector<int> >  _mapSoft2clause;
public:

    virtual ~VirtualMAXSAT();

    virtual unsigned int newSoftVar(bool value, t_weight weight) = 0;

    virtual bool isSoft(unsigned int var) = 0;

    virtual void setVarSoft(unsigned int var, bool value, t_weight weight=1) = 0;

    virtual t_weight getCost() = 0;


    void setIsWeightedVerif() {  // TODO : remplacer par  mapWeight2Assum
        _isWeighted = true;
    }
    virtual bool isWeightedVerif() { // TODO : remplacer par  mapWeight2Assum
        return _isWeighted;
    }

    virtual bool isWeighted() = 0;

    unsigned int nInputVars=0;
    void setNInputVars(unsigned int nInputVars) {
        this->nInputVars=nInputVars;
    }

    int addWeightedClause(std::vector<int> clause, t_weight weight) {
        // If it's a unit clause and its literal doesn't exist as a soft var already, add soft variable
        if(clause.size() == 1) {
            // add weight to the soft var
            setVarSoft(abs(clause[0]), clause[0] > 0, weight);

            // Return instantly instead of adding a new var at the end because the soft var represents the unit clause anyway.
            return clause[0];
        }

        // Soft clauses are "hard" clauses with a soft var at the end. Create said soft var for our clause.
        int r = static_cast<int>(newSoftVar(true, weight));
        clause.push_back( -r );
        addClause(clause);
        clause.pop_back();

        assert(r > 0);
        if(r >= _mapSoft2clause.size()) {
            _mapSoft2clause.resize(r+1);
        }
        _mapSoft2clause[r] = clause;
        //mapSoft2clause[r] = clause;

        return r;
    }


/* Removed for lincs
   std::string savePourTest_file;
   bool parse(const std::string& filePath) {
       auto gz = gzopen( filePath.c_str(), "rb");

       savePourTest_file = filePath;
       StreamBuffer in(gz);
       t_weight weightForHardClause = -1;

       if(*in == EOF) {
           return false;
       }

       std::vector < std::tuple < std::vector<int>, t_weight> > softClauses;

       for(;;) {
           skipWhitespace(in);

           if(*in == EOF) {
               break;
           }

           if(*in == 'c') {
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
           } else {
               t_weight weight = parseWeight(in);
               std::vector<int> clause = readClause(in);

               if(weight == weightForHardClause) {
                   addClause(clause);
               } else {
                   // If it is a soft clause, we have to save it to add it once we are sure we know the total number of variables.
                   softClauses.push_back({clause, weight});
               }
           }
       }

       setNInputVars(nVars());
       for(auto & [clause, weight]: softClauses) {
           addWeightedClause(clause, weight);
       }

       gzclose(gz);
       return true;
    }

private :


   std::vector<int> readClause(StreamBuffer &in) {
       std::vector<int> clause;

       for (;;) {
           int lit = parseInt(in);

           if (lit == 0)
               break;
           clause.push_back(lit);
           while( abs(lit) > nVars()) {
               newVar();
           }
       }

       return clause;
   }



*/  // Removed for lincs
};
inline VirtualMAXSAT::~VirtualMAXSAT() {}

#endif // VIRTUALMAXSAT_H
