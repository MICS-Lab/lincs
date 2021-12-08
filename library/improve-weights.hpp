// Copyright 2021 Vincent Jacques

#ifndef IMPROVE_WEIGHTS_HPP_
#define IMPROVE_WEIGHTS_HPP_

#include "problem.hpp"


namespace ppl {

// Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
void improve_weights(Models<Host>*);

}  // namespace ppl

#endif  // IMPROVE_WEIGHTS_HPP_
