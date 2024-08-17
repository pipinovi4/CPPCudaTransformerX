#ifndef FLOAT16_TPP
#define FLOAT16_TPP

// Default constructor, initializes the value to 0 (equivalent to 0.0 in float16)
inline float_16::float_16() : value(0) {}

// Constructor that converts a 32-bit float to 16-bit float and stores it
inline float_16::float_16(const float f) : value(float32_to_float16(f)) {}

// Conversion operator that converts the 16-bit float back to 32-bit float
inline float_16::operator float() const {
    return float16_to_float32(value);
}

// Addition operator using default libraries of the C++ language
// This method uses basic bit manipulation to perform addition of two 16-bit floats.
inline float_16 float_16::operator+(const float_16& other) const {
    // Extract components of the first operand
    uint16_t sign1 = value & 0x8000;  // Extract the sign bit
    uint16_t exponent1 = (value >> 10) & 0x1F;  // Extract the exponent
    uint16_t mantissa1 = value & 0x03FF;  // Extract the mantissa

    // Extract components of the second operand
    const uint16_t sign2 = other.value & 0x8000;  // Extract the sign bit
    uint16_t exponent2 = (other.value >> 10) & 0x1F;  // Extract the exponent
    uint16_t mantissa2 = other.value & 0x03FF;  // Extract the mantissa

    // Handle zero and special cases
    if (exponent1 == 0) mantissa1 = 0;  // Zero or subnormal
    if (exponent2 == 0) mantissa2 = 0;  // Zero or subnormal

    // Normalize subnormals to have implicit leading 1 bit
    if (exponent1 != 0) mantissa1 |= 0x0400;
    if (exponent2 != 0) mantissa2 |= 0x0400;

    // Align exponents by shifting the mantissa of the smaller exponent
    if (exponent1 > exponent2) {
        mantissa2 >>= (exponent1 - exponent2);
        exponent2 = exponent1;
    } else if (exponent2 > exponent1) {
        mantissa1 >>= (exponent2 - exponent1);
        exponent1 = exponent2;
    }

    // Add or subtract mantissas based on sign
    uint16_t result_mantissa;
    if (sign1 == sign2) {
        result_mantissa = mantissa1 + mantissa2;  // Same sign, add mantissas
    } else {
        if (mantissa1 >= mantissa2) {
            result_mantissa = mantissa1 - mantissa2;  // Subtract mantissas
        } else {
            result_mantissa = mantissa2 - mantissa1;
            sign1 = sign2;  // Result takes the sign of the larger magnitude
        }
    }

    // Normalize the result mantissa and adjust the exponent
    if (result_mantissa & 0x0800) {  // Overflow in mantissa
        result_mantissa >>= 1;
        exponent1++;
    } else {
        while ((result_mantissa & 0x0400) == 0 && exponent1 > 0) {
            result_mantissa <<= 1;
            exponent1--;
        }
    }

    // Handle rounding
    if (result_mantissa & 0x0003) {  // Check if rounding is needed
        result_mantissa += 0x0002;  // Round to nearest even
        if (result_mantissa & 0x0800) {  // Handle overflow due to rounding
            result_mantissa >>= 1;
            exponent1++;
        }
    }

    // Handle overflow (result is too large for float16)
    if (exponent1 >= 0x1F) {
        return float_16(sign1 | 0x7C00);  // Return infinity with the correct sign
    }

    // Handle underflow (result is too small for float16)
    if (exponent1 <= 0) {
        return float_16(sign1);  // Return zero with the correct sign
    }

    // Reconstruct the result
    const uint16_t result_value = sign1 | (exponent1 << 10) | (result_mantissa & 0x03FF);

    // Return the result
    return float_16(result_value);
}

// Subtraction operator using default 32-bit float arithmetic
inline float_16 float_16::operator-(const float_16& other) const {
    return float_16(static_cast<float>(*this) - static_cast<float>(other));
}

// Multiplication operator using default 32-bit float arithmetic
inline float_16 float_16::operator*(const float_16& other) const {
    return float_16(static_cast<float>(*this) * static_cast<float>(other));
}

// Division operator using default 32-bit float arithmetic
inline float_16 float_16::operator/(const float_16& other) const {
    return float_16(static_cast<float>(*this) / static_cast<float>(other));
}

// Compound assignment operators using default arithmetic
inline float_16& float_16::operator+=(const float_16& other) {
    *this = *this + other;
    return *this;
}

inline float_16& float_16::operator-=(const float_16& other) {
    *this = *this - other;
    return *this;
}

inline float_16& float_16::operator*=(const float_16& other) {
    *this = *this * other;
    return *this;
}

inline float_16& float_16::operator/=(const float_16& other) {
    *this = *this / other;
    return *this;
}

// Overloaded comparison operators
inline bool float_16::operator==(const float_16& other) const {
    return static_cast<float>(*this) == static_cast<float>(other);
}

inline bool float_16::operator!=(const float_16& other) const {
    return !(*this == other);
}

inline bool float_16::operator<(const float_16& other) const {
    return static_cast<float>(*this) < static_cast<float>(other);
}

inline bool float_16::operator<=(const float_16& other) const {
    return *this < other || *this == other;
}

inline bool float_16::operator>(const float_16& other) const {
    return !(*this <= other);
}

inline bool float_16::operator>=(const float_16& other) const {
    return !(*this < other);
}

// Overloaded increment and decrement operators
inline float_16& float_16::operator++() {
    *this += float_16(1.0f);  // Increment by 1.0 in float16 representation
    return *this;
}

inline float_16 float_16::operator++(int) {
    float_16 temp = *this;
    ++(*this);  // Increment by 1.0 in float16 representation
    return temp;
}

inline float_16& float_16::operator--() {
    *this -= float_16(1.0f);  // Decrement by 1.0 in float16 representation
    return *this;
}

inline float_16 float_16::operator--(int) {
    float_16 temp = *this;
    --(*this);  // Decrement by 1.0 in float16 representation
    return temp;
}

// Overloaded unary operators
inline float_16 float_16::operator+() const {
    return *this;  // Unary plus, no change
}

inline float_16 float_16::operator-() const {
    return float_16(-static_cast<float>(*this));  // Unary minus, negates the value
}

// Overloaded assignment operator
inline float_16& float_16::operator=(const float_16& other) {
    if (this != &other) {
        value = other.value;
    }
    return *this;
}

// Conversion functions
// Converts a 32-bit float to a 16-bit float
inline uint16_t float_16::float32_to_float16(const float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint16_t sign = (bits >> 16) & 0x8000;  // Extract sign bit
    const uint16_t exponent = ((bits >> 23) & 0xff) - 127 + 15;  // Adjust exponent
    uint16_t mantissa = (bits >> 13) & 0x3ff;  // Extract mantissa

    if (exponent <= 0) {
        // Subnormal
        if (exponent < -10) {
            // Too small, return zero
            return sign;
        }
        mantissa = (mantissa | 0x400) >> (1 - exponent);
        return sign | mantissa;
    } else if (exponent >= 0x1f) {
        // Inf or NaN
        return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    }

    return sign | (exponent << 10) | mantissa;
}

// Converts a 16-bit float to a 32-bit float
inline float float_16::float16_to_float32(const uint16_t h) {
    const uint32_t sign = (h & 0x8000) << 16;  // Extract sign bit
    uint32_t exponent = (h & 0x7c00) >> 10;  // Extract exponent
    uint32_t mantissa = (h & 0x03ff) << 13;  // Extract mantissa

    if (exponent == 0) {
        // Subnormal or zero
        if (mantissa == 0) {
            return bit_cast<float>(sign);  // Zero
        } else {
            // Normalize the subnormal number
            exponent = 127 - 15;
            while ((mantissa & 0x400000) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3fffff;
            exponent += 127 - 15;
        }
    } else if (exponent == 0x1f) {
        // Inf or NaN
        exponent = 255;
    } else {
        exponent += 127 - 15;
    }

    const uint32_t bits = sign | (exponent << 23) | mantissa;
    return bit_cast<float>(bits);
}

// Overloaded output stream operator
inline std::ostream& operator<<(std::ostream& os, const float_16& f16) {
    // Convert the float_16 value to a float
    const float float_value = float_16::float16_to_float32(f16.value);

    // Output the float value to the provided output stream
    os << float_value;

    return os;
}

#endif // FLOAT16_TPP
