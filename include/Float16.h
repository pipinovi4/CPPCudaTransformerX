#ifndef FLOAT16_H
#define FLOAT16_H

#pragma once
#include <iostream>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

/**
 * @class float_16
 * @brief A class that represents a 16-bit floating point number, often referred to as half-precision float.
 *
 * This class provides methods for converting between 32-bit (single-precision) and 16-bit (half-precision)
 * floating point numbers, as well as overloaded operators for basic arithmetic, comparison, and stream output.
 * The implementation is optimized using hardware-specific instructions where possible.
 */
class float_16 {
public:
    int16_t value; ///< The underlying 16-bit representation of the half-precision float.

    /**
     * @brief Default constructor. Initializes the value to 0 (half-precision representation of 0.0f).
     */
    inline float_16();

    /**
     * @brief Constructor from a 32-bit float.
     * @param f The 32-bit float to be converted to half-precision.
     */
    inline explicit float_16(float f);

    /**
     * @brief Conversion operator to a 32-bit float.
     * @return The half-precision float as a 32-bit float.
     */
    inline explicit operator float() const;

    // Overloaded operators for basic arithmetic

    /**
     * @brief Addition operator for half-precision floats.
     * @param other The other half-precision float to add.
     * @return The result of the addition.
     */
    inline float_16 operator+(const float_16& other) const;

    /**
     * @brief Subtraction operator for half-precision floats.
     * @param other The other half-precision float to subtract.
     * @return The result of the subtraction.
     */
    inline float_16 operator-(const float_16& other) const;

    /**
     * @brief Multiplication operator for half-precision floats.
     * @param other The other half-precision float to multiply.
     * @return The result of the multiplication.
     */
    inline float_16 operator*(const float_16& other) const;

    /**
     * @brief Division operator for half-precision floats.
     * @param other The other half-precision float to divide by.
     * @return The result of the division.
     */
    inline float_16 operator/(const float_16& other) const;

    // Compound assignment operators

    /**
     * @brief Compound addition operator.
     * @param other The other half-precision float to add.
     * @return Reference to the updated object.
     */
    inline float_16& operator+=(const float_16& other);

    /**
     * @brief Compound subtraction operator.
     * @param other The other half-precision float to subtract.
     * @return Reference to the updated object.
     */
    inline float_16& operator-=(const float_16& other);

    /**
     * @brief Compound multiplication operator.
     * @param other The other half-precision float to multiply.
     * @return Reference to the updated object.
     */
    inline float_16& operator*=(const float_16& other);

    /**
     * @brief Compound division operator.
     * @param other The other half-precision float to divide by.
     * @return Reference to the updated object.
     */
    inline float_16& operator/=(const float_16& other);

    // Comparison operators

    /**
     * @brief Equality operator.
     * @param other The other half-precision float to compare.
     * @return True if the two values are equal, false otherwise.
     */
    inline bool operator==(const float_16& other) const;

    /**
     * @brief Inequality operator.
     * @param other The other half-precision float to compare.
     * @return True if the two values are not equal, false otherwise.
     */
    inline bool operator!=(const float_16& other) const;

    /**
     * @brief Less-than operator.
     * @param other The other half-precision float to compare.
     * @return True if this value is less than the other value, false otherwise.
     */
    inline bool operator<(const float_16& other) const;

    /**
     * @brief Less-than-or-equal-to operator.
     * @param other The other half-precision float to compare.
     * @return True if this value is less than or equal to the other value, false otherwise.
     */
    inline bool operator<=(const float_16& other) const;

    /**
     * @brief Greater-than operator.
     * @param other The other half-precision float to compare.
     * @return True if this value is greater than the other value, false otherwise.
     */
    inline bool operator>(const float_16& other) const;

    /**
     * @brief Greater-than-or-equal-to operator.
     * @param other The other half-precision float to compare.
     * @return True if this value is greater than or equal to the other value, false otherwise.
     */
    inline bool operator>=(const float_16& other) const;

    // Increment and decrement operators

    /**
     * @brief Prefix increment operator.
     * @return Reference to the incremented object.
     */
    inline float_16& operator++();

    /**
     * @brief Postfix increment operator.
     * @return The value before incrementing.
     */
    inline float_16 operator++(int);

    /**
     * @brief Prefix decrement operator.
     * @return Reference to the decremented object.
     */
    inline float_16& operator--();

    /**
     * @brief Postfix decrement operator.
     * @return The value before decrementing.
     */
    inline float_16 operator--(int);

    // Unary operators

    /**
     * @brief Unary plus operator.
     * @return The positive value.
     */
    inline float_16 operator+() const;

    /**
     * @brief Unary negation operator.
     * @return The negated value.
     */
    inline float_16 operator-() const;

    // Assignment operator

    /**
     * @brief Assignment operator.
     * @param other The other half-precision float to assign.
     * @return Reference to the assigned object.
     */
    inline float_16& operator=(const float_16& other);

    // Output stream operator

    /**
     * @brief Stream output operator for float_16.
     * @param os The output stream.
     * @param f16 The half-precision float to output.
     * @return Reference to the output stream.
     */
    inline friend std::ostream& operator<<(std::ostream& os, const float_16& f16);

    // Static utility functions for conversion

    /**
     * @brief Converts a 32-bit float to a 16-bit float.
     * @param f The 32-bit float to convert.
     * @return The 16-bit half-precision representation.
     */
    static inline uint16_t float32_to_float16(float f);

    /**
     * @brief Converts a 16-bit float to a 32-bit float.
     * @param h The 16-bit float to convert.
     * @return The 32-bit single-precision representation.
     */
    static inline float float16_to_float32(uint16_t h);
};

#include "../src/Float16.tpp"

#endif // FLOAT16_H
