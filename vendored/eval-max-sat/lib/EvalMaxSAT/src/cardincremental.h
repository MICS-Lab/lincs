#ifndef CARDINCREMENTAL_H
#define CARDINCREMENTAL_H

#include "virtualcard.h"
#include <optional>
#include <cassert>
#include <memory>
#include <deque>
#include <iostream>

#include "lazyvariable.h"


class CardIncremental_Lazy : public VirtualCard {

   struct TotTree {
        // Each non-leaf node can have multiple vars who have impliquants
        std::vector< std::shared_ptr<LazyVariable> > lazyVars;
        unsigned nof_input; // Number of literals under the node
        std::shared_ptr<TotTree> left;
        std::shared_ptr<TotTree> right;

        void print(std::ostream& os, bool first=true) const {
            if(left == nullptr && right == nullptr) {
                assert( lazyVars.size() == 1);
                if(!first)
                    os << ", ";
                os << lazyVars[0];
            } else {
                if(left != nullptr) {
                    left->print(os, first);
                }
                if(right != nullptr) {
                    right->print(os, false);
                }
            }
        }


        std::vector<int> getClause() {
            std::vector<int> clause;
            getClause(clause);
            return clause;
        }

   private:
       void getClause(std::vector<int> &clause) {
           if(left == nullptr && right == nullptr) {
               assert(lazyVars.size() == 1);
               clause.push_back( lazyVars[0]->get() );
           } else {
               if(left != nullptr) {
                   left->getClause(clause);
               }
               if(right != nullptr) {
                   right->getClause(clause);
               }
           }
       }

    };

    std::shared_ptr<TotTree> _tree;
    unsigned int _maxVars; // Max number of literals in a tree, ignoring k-simplification
public:

    virtual std::vector<int> getClause() override {
        return _tree->getClause();
    }


    void print(std::ostream& os) const override {
        os << "[";
        _tree->print(os, true);
        os << "]";
    }

    unsigned int size() const override {
        return _maxVars;
    }

    void add(const std::vector<int>& clause);


    virtual int atMost(unsigned int k) override {
        // if the bound is bigger or equal to the current bound
        if( k >= _maxVars ) {
            return 0;
        }

        // Increase node (add vars) if possible and needed
        if( k >= _tree->lazyVars.size() ) {
            increase(_tree, k);
        }
        assert(k < _tree->lazyVars.size());

        // Return the soft var corresponding to a cardinality constraint from the tree with a bound of k
        return -_tree->lazyVars[k]->get();
    }

    CardIncremental_Lazy(VirtualSAT * solver, const std::vector<int>& clause, unsigned int bound=1);



private:

    void increase(std::shared_ptr<TotTree> tree, unsigned newBound)
    {
        unsigned kmin = std::min(newBound + 1, tree->nof_input);

        // Each new var in a parent node must have enough literals under it to make up for its representation in the unary number ;
        if (tree->lazyVars.size() >= kmin) // In most cases, only continue if the node has been affected by k-simplification
            return;                        // and its nof_input is smaller than the number of vars, leaving room for increase.

        increase   (tree->left, newBound);
        increase   (tree->right, newBound);
        increase_ua( tree->lazyVars, tree->left->lazyVars, tree->right->lazyVars, kmin);
    }

    void increase_ua( std::vector< std::shared_ptr<LazyVariable> >& ogVars, std::vector< std::shared_ptr<LazyVariable> >& aVars, std::vector< std::shared_ptr<LazyVariable> >& bVars, unsigned rhs);


    void new_ua( std::vector< std::shared_ptr<LazyVariable> >& ogVars, unsigned rhs, std::vector< std::shared_ptr<LazyVariable> >& aVars, std::vector< std::shared_ptr<LazyVariable> >& bVars);


    friend class VirtualSAT;
};


#endif // CARDINCREMENTAL_H
