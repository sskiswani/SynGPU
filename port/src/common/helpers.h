#ifndef PORT_HELPERS_H
#define PORT_HELPERS_H

#include <iostream>
#include <iomanip>
#include <stdlib.h>

#ifdef __GPU_BUILD__
#include "cuda_utils.h"
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

template<typename T>
class TArray2 {
  public:
#ifdef __GPU_BUILD__
    friend class CUSynfire;
#endif

    friend std::ostream& operator<< (std::ostream& stream, const TArray2<T>& array) {
        for(int y = 0, i =0; y < array._rows; ++y) {
            stream << "  [";
            for(int x = 0; x < array._cols; ++x, ++i) {
                stream << std::setprecision(3) << " " << std::setw(4) <<  array._data[i] << " ";
            }
            stream << "]" << std::endl;
        }

        return stream;
    }

    // typedefs
    typedef T value_type;
    typedef value_type *iterator;
    typedef const value_type *const_iterator;

    TArray2( int cols = 1, int rows = 1 )
            : _cols(cols), _rows(rows) {
        _data = new T[_cols * _rows];
    }

    ~TArray2() {
        delete[] _data;
    }

#ifdef __GPU_BUILD__
    TArray2<T> *CopyToDevice() {
        TArray2<T> *deviceArray;
        T *d_data;

        // Copy the TArray to the device.
        HANDLE_ERROR(cudaMalloc(&deviceArray, sizeof(TArray2<T>)));
        HANDLE_ERROR(cudaMemcpy(deviceArray, this, sizeof(TArray2<T>), cudaMemcpyHostToDevice));

        // Copy the elements array to the device.
        HANDLE_ERROR(cudaMalloc((void **) &(d_data), sizeof(T) * (_rows * _cols)));

        // Link data elements to deviceA
        HANDLE_ERROR(cudaMemcpy(&(deviceArray->_data), &d_data, sizeof(T *), cudaMemcpyHostToDevice));

        // Copy the data over.
        HANDLE_ERROR(cudaMemcpy(d_data, _data, sizeof(T) * (_rows * _cols), cudaMemcpyHostToDevice));

        return deviceArray;
    }
#endif


    // Accessors
    CUDA_CALLABLE inline int Rows() const { return _rows; }

    CUDA_CALLABLE inline int Columns() const { return _cols; }

    CUDA_CALLABLE inline size_t Bytes() const { return (_rows * _cols) * sizeof(T); }

    // STL style iterators
    CUDA_CALLABLE inline const_iterator begin() const { return _data; }

    CUDA_CALLABLE inline const_iterator end() const { return _data + (_rows * _cols); }

    CUDA_CALLABLE inline iterator begin() { return _data; }

    CUDA_CALLABLE inline iterator end() { return _data + (_rows * _cols); }

    CUDA_CALLABLE inline iterator row( int y ) { return _data + (y * _cols); }

    // Operators
    CUDA_CALLABLE inline T &operator[]( const int i ) { return _data[i]; }

    CUDA_CALLABLE inline T &operator()( int i ) { return _data[i]; }

    CUDA_CALLABLE inline T &operator()( int x, int y ) { return _data[x + y * _cols]; }

    T *_data;
  private:
    int _cols, _rows;
};

#endif //PORT_HELPERS_H
