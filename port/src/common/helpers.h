#ifndef PORT_HELPERS_H
#define PORT_HELPERS_H

#include <stdlib.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

template<typename T>
class TArray2 {
  public:
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

  private:
    int _cols, _rows;
    T *_data;
};

#endif //PORT_HELPERS_H
