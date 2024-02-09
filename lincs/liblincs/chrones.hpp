// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__CHRONES_HPP
#define LINCS__CHRONES_HPP

#ifdef LINCS_HAS_CHRONES
#include <chrones.hpp>
#else
#define CHRONABLE(name)
#define CHRONE(...)
#endif

#endif  // LINCS__CHRONES_HPP
