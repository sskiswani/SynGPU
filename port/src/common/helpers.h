#ifndef PORT_HELPERS_H
#define PORT_HELPERS_H

#include <stdlib.h>
#include <iostream>

template<typename T>
class TArray2 {
  public:
    // typedefs
    typedef T value_type;
    typedef value_type *iterator;
    typedef const value_type *const_iterator;

    TArray2( int cols = 1, int rows = 1 )
            : _cols(cols), _rows(rows), _data(0) {
        _data = new T[_cols * _rows];
    }

    ~TArray2() {
        delete[] _data;
    }


    // Accessors
    inline int Rows() const { return _rows; }

    inline int Columns() const { return _cols; }

    inline size_t Bytes() const { return (_rows * _cols) * sizeof(T); }

    // STL style iterators
    inline const_iterator begin() const { return _data; }

    inline const_iterator end() const { return _data + (_rows * _cols); }

    inline iterator begin() { return _data; }

    inline iterator end() { return _data + (_rows * _cols); }

    inline iterator row( int y ) { return _data + (y * _cols); }

    // Operators
    inline T &operator[]( const int i ) { return _data[i]; }

    inline T &operator()( int i ) { return _data[i]; }

    inline T &operator()( int x, int y ) { return _data[x + y * _cols]; }

  private:
    int _cols, _rows;
    T *_data;
};

#endif //PORT_HELPERS_H
