#ifndef FLOAT16_H
#define FLOAT16_H

#pragma once
#include <iostream>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

class float_16 {
public:
    int16_t value;

    // Default constructor
    inline float_16();

    // Constructor from float
    inline float_16(float f);

    // Conversion to float
    inline operator float() const;

    // Overloaded operators for basic arithmetic
    inline float_16 operator+(const float_16& other) const;
    inline float_16 operator-(const float_16& other) const;
    inline float_16 operator*(const float_16& other) const;
    inline float_16 operator/(const float_16& other) const;

    inline float_16& operator+=(const float_16& other);
    inline float_16& operator-=(const float_16& other);
    inline float_16& operator*=(const float_16& other);
    inline float_16& operator/=(const float_16& other);

    // Overloaded comparison operators
    inline bool operator==(const float_16& other) const;
    inline bool operator!=(const float_16& other) const;
    inline bool operator<(const float_16& other) const;
    inline bool operator<=(const float_16& other) const;
    inline bool operator>(const float_16& other) const;
    inline bool operator>=(const float_16& other) const;

    // Overloaded increment and decrement operators
    inline float_16& operator++();
    inline float_16 operator++(int);
    inline float_16& operator--();
    inline float_16 operator--(int);

    // Overloaded unary operators
    inline float_16 operator+() const;
    inline float_16 operator-() const;

    // Overloaded assignment operator
    inline float_16& operator=(const float_16& other);

    // Declare the friend function for the output stream operator
    inline friend std::ostream& operator<<(std::ostream& os, const float_16& f16);

    static inline uint16_t float32_to_float16(float f);
    static inline float float16_to_float32(uint16_t h);
};

#include "../src/Float16.tpp"

#endif // FLOAT16_H
