#ifndef BUFFER1D_H
#define BUFFER1D_H

#include <vector>
#include <exception>
#include <cstring>
#include <memory>
#include <algorithm>
#include <iostream>


template <typename T>
struct Buffer1D{

    T* buf;
    std::size_t n;

    Buffer1D(std::size_t _n = 0);
    
    Buffer1D(const Buffer1D& rhs);
    
    Buffer1D(Buffer1D && rhs);
    
    Buffer1D<T>& operator=(const Buffer1D & rhs);
    
    Buffer1D<T>& operator=(Buffer1D && rhs);

    T& operator[](std::size_t i) const;

    ~Buffer1D();
};


template <typename T> Buffer1D<T>::Buffer1D(std::size_t _n):n(_n){
    buf = new T[n];
}

template <typename T> Buffer1D<T>::Buffer1D(const Buffer1D<T>& rhs): n(rhs.n){
    buf = new T[n];
    memcpy(buf, rhs.buf, n*sizeof(T));
}

template <typename T> Buffer1D<T>::Buffer1D(Buffer1D<T> && rhs): n(rhs.n), buf(rhs.buf){
    rhs.buf = nullptr;
}

template <typename T> 
Buffer1D<T>& Buffer1D<T>::operator=(const Buffer1D<T>& rhs){
    if(&rhs == this)
        return *this;
    if(n != rhs.n){
        this->n = rhs.n;
        delete[] buf;
        buf = new T[rhs.n];
    }

    memcpy(buf, rhs.buf, n*sizeof(T));
    return *this;
}

template <typename T> 
Buffer1D<T>& Buffer1D<T>::operator=(Buffer1D<T>&& rhs){
    if(&rhs == this)
        return *this;
    n = rhs.n;
    delete[] buf;
    buf = rhs.buf;
    rhs.buf = nullptr;
    return *this;
}

template <typename T> 
T& Buffer1D<T>::operator[] (std::size_t i) const {
    return buf[i];
}

template <typename T> Buffer1D<T>::~Buffer1D(){
    delete[] buf;
}

#endif //BUFFER1D_H