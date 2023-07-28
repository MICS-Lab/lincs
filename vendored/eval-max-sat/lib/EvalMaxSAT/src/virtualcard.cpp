
#include "virtualcard.h"

#include "virtualsat.h"

std::ostream& operator<<(std::ostream& os, const VirtualCard& dt) {
    dt.print(os);
    return os;
}


VirtualCard::~VirtualCard() {

}
