#include "cardincremental.h"

#include "virtualsat.h"


// Create a tree with each literal from the passed clause as a leaf.
CardIncremental_Lazy::CardIncremental_Lazy(VirtualSAT * solver, const std::vector<int>& clause, unsigned int bound)
    : VirtualCard (solver, clause, bound), _maxVars( clause.size())
{
    std::deque<std::shared_ptr<TotTree>> nqueue;

    // Create the leafs and store them in a queue
    for ( unsigned i = 0; i < _maxVars; ++i) {
        std::shared_ptr<TotTree> node = std::make_shared<TotTree>();

        node->lazyVars.push_back( LazyVariable::encapsulate( clause[i]));
        node->nof_input = 1;
        node->left      = nullptr;
        node->right     = 0;

        nqueue.push_back( node);
    }

    // Create non-leaf nodes from the bottom-up by starting from the beginning of the queue
    while (nqueue.size() > 1) {
        auto l = nqueue.front();
        nqueue.pop_front();
        auto r = nqueue.front();
        nqueue.pop_front();

        auto node = std::make_shared<TotTree>();
        node->nof_input = l->nof_input + r->nof_input;
        node->left      = l;
        node->right     = r;

        // Bound is the RHS. No need to represent more than RHS + 1 because of k-simplification
        unsigned kmin = std::min(bound + 1, node->nof_input);

        node->lazyVars.resize( kmin);
        for (unsigned i = 0; i < kmin; ++i)
            node->lazyVars[i] = LazyVariable::newVar( solver);

        new_ua( node->lazyVars, kmin, l->lazyVars, r->lazyVars);
        nqueue.push_back(node);
    }

    _tree = nqueue.front();
}

// Add a node in the tree, adding it at the 'right' of the previous root and adding a new root.
void CardIncremental_Lazy::add(const std::vector<int>& clause) {
    CardIncremental_Lazy tb(solver, clause, _tree->lazyVars.size() - 1);

    unsigned n    = _tree->nof_input + tb._tree->nof_input;
    unsigned kmin = n;
    if( _tree->lazyVars.size() < n)
        kmin = _tree->lazyVars.size();


    std::shared_ptr<TotTree> tree = std::make_shared<TotTree>();
    tree->nof_input = n;
    tree->left      = _tree;
    tree->right     = tb._tree;

    tree->lazyVars.resize( kmin);
    for (unsigned i = 0; i < kmin; ++i)
        tree->lazyVars[i] = LazyVariable::newVar( solver);

    new_ua( tree->lazyVars, kmin, _tree->lazyVars, tb._tree->lazyVars);

    _maxVars += clause.size();
    _tree = tree;
}

// Add implications (like new_ua) but for an existing node that already has some, starting from the end.
void CardIncremental_Lazy::increase_ua( std::vector< std::shared_ptr<LazyVariable> >& ogVars, std::vector< std::shared_ptr<LazyVariable> >& aVars, std::vector< std::shared_ptr<LazyVariable> >& bVars, unsigned rhs)
{
    unsigned last = ogVars.size();

    for (unsigned i = last; i < rhs; ++i)
        ogVars.push_back( LazyVariable::newVar( solver) );

    unsigned maxj = std::min(rhs, (unsigned)bVars.size());
    for (unsigned j = last; j < maxj; ++j) {
        ogVars[j]->addImpliquant( {bVars[j]});
    }

    unsigned maxi = std::min(rhs, (unsigned)aVars.size());
    for (unsigned i = last; i < maxi; ++i) {
        //addUnitClause({-aVars[i], ogVars[i]});
        ogVars[i]->addImpliquant( {aVars[i]});
    }

    for (unsigned i = 1; i <= maxi; ++i) {
        unsigned maxj = std::min(rhs - i, (unsigned)bVars.size());
        unsigned minj = std::max((int)last - (int)i + 1, 1);
        for (unsigned j = minj; j <= maxj; ++j) {
            ogVars[ i + j - 1]->addImpliquant( {aVars[ i - 1], bVars[ j - 1]});
        }
    }
}

// Add the necessary implications for a new node, starting from scratch.
void CardIncremental_Lazy::new_ua( std::vector< std::shared_ptr<LazyVariable> >& ogVars, unsigned rhs, std::vector< std::shared_ptr<LazyVariable> >& aVars, std::vector< std::shared_ptr<LazyVariable> >& bVars)
{
    // Creates a direct correspondance between an ogVar and a bVar of the same index
    unsigned kmin = std::min(rhs, (unsigned)bVars.size());
    for (unsigned j = 0; j < kmin; ++j) {
        ogVars[j]->addImpliquant( {bVars[j]});
    }

    // Same as above ; if aVar[index] is true, then ogVar[index] must be true as well
    kmin = std::min(rhs, (unsigned)aVars.size());
    for (unsigned i = 0; i < kmin; ++i) {
        ogVars[i]->addImpliquant( {aVars[i]});
    }

    // Handles the addition cases. Per example, if aVar[0] is true and bVar[2] is true, then ogVar[3] must be true.
    // Refer to a Totalizer Encoding tree.
    for (unsigned i = 1; i <= kmin; ++i) {
        unsigned minj = std::min(rhs - i, (unsigned)bVars.size());
        for (unsigned j = 1; j <= minj; ++j) {
            ogVars[ i + j - 1]->addImpliquant( {aVars[ i - 1], bVars[ j - 1]});
        }
    }
}

