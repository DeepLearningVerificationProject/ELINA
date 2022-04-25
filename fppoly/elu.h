//
// Created by Kirby Linvill on 4/25/22.
//

#ifndef ELINA_ELU_H
#define ELINA_ELU_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

inline double elu (double x) {
    if (x < 0)
        return exp(x) - 1;
    else
        return x;
}

#ifdef __cplusplus
}
#endif

#endif //ELINA_ELU_H
