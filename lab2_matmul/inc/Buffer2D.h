#ifndef BUFFER2D_H
#define BUFFER2D_H

#include "./Buffer1D.h"

template <typename T>
struct Buffer2D{

    Buffer1D<T> buf;
    int n, m;


    Buffer2D(int _n = 0, int _m = 0);
    
    Buffer2D(const Buffer2D<T> &) = default;
    
    Buffer2D(Buffer2D<T> &&) = default;
    
    Buffer2D<T>& operator=(const Buffer2D<T> &) = default;
    
    Buffer2D<T>& operator=(Buffer2D<T> &&) = default;

    T& at(std::size_t i, std::size_t j) const;

    ~Buffer2D() = default;
};


template <typename T> Buffer2D<T>::Buffer2D(int _n, int _m):m(_m), n(_n), buf(_n*_m){}


template <typename T>
T& Buffer2D<T>::at(std::size_t i, std::size_t j) const {
    return buf[i*m+j];
}

#endif //BUFFER2D_H