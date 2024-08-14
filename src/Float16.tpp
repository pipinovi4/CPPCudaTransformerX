#ifndef FLOAT16_TPP
#define FLOAT16_TPP

// Constructors
inline float_16::float_16() : value(0) {}

inline float_16::float_16(const float f) : value(float32_to_float16(f)) {}

// Conversion to float
inline float_16::operator float() const {
    return float16_to_float32(value);
}

// Using immintrin library for SIMD operations (This is not work on my Architecture)
// inline float_16 float_16::operator+(const float_16& other) const {
//     // Load the values into SIMD registers
//     __m128i a = _mm_set1_epi16(value);   // Broadcasts the 16-bit value across all elements
//     __m128i b = _mm_set1_epi16(other.value);
//
//     // Extract the sign, exponent, and mantissa using SIMD
//     __m128i sign_mask = _mm_set1_epi16(0x8000);
//     __m128i exp_mask = _mm_set1_epi16(0x7C00);
//     __m128i mantissa_mask = _mm_set1_epi16(0x03FF);
//
//     __m128i sign_a = _mm_and_si128(a, sign_mask);
//     __m128i sign_b = _mm_and_si128(b, sign_mask);
//
//     __m128i exponent_a = _mm_srli_epi16(_mm_and_si128(a, exp_mask), 10);
//     __m128i exponent_b = _mm_srli_epi16(_mm_and_si128(b, exp_mask), 10);
//
//     __m128i mantissa_a = _mm_and_si128(a, mantissa_mask);
//     __m128i mantissa_b = _mm_and_si128(b, mantissa_mask);
//
//     // Normalize mantissas if exponents are non-zero
//     mantissa_a = _mm_blendv_epi8(_mm_or_si128(mantissa_a, _mm_set1_epi16(0x0400)), mantissa_a, _mm_cmpeq_epi16(exponent_a, _mm_setzero_si128()));
//     mantissa_b = _mm_blendv_epi8(_mm_or_si128(mantissa_b, _mm_set1_epi16(0x0400)), mantissa_b, _mm_cmpeq_epi16(exponent_b, _mm_setzero_si128()));
//
//     // Align exponents by shifting mantissas of the smaller exponent
//     __m128i exp_diff = _mm_sub_epi16(exponent_a, exponent_b);
//     __m128i shift_mask = _mm_cmpgt_epi16(exponent_a, exponent_b);
//
//     mantissa_a = _mm_srli_epi16(mantissa_a, _mm_and_si128(exp_diff, shift_mask));
//     mantissa_b = _mm_srli_epi16(mantissa_b, _mm_andnot_si128(shift_mask, exp_diff));
//     exponent_a = _mm_max_epi16(exponent_a, exponent_b);
//
//     // Perform the addition or subtraction of mantissas based on sign
//     __m128i result_mantissa = _mm_add_epi16(mantissa_a, mantissa_b);
//     __m128i result_sign = _mm_blendv_epi8(sign_a, sign_b, _mm_cmpgt_epi16(mantissa_b, mantissa_a));
//
//     // Handle normalization and rounding (simplified for SIMD)
//     __m128i overflow = _mm_and_si128(result_mantissa, _mm_set1_epi16(0x0800));
//     result_mantissa = _mm_srli_epi16(result_mantissa, _mm_testz_si128(overflow, overflow));
//     exponent_a = _mm_add_epi16(exponent_a, _mm_testz_si128(overflow, overflow));
//
//     // Pack the result back into a float_16 structure
//     __m128i result_value = _mm_or_si128(result_sign, _mm_slli_epi16(exponent_a, 10));
//     result_value = _mm_or_si128(result_value, _mm_and_si128(result_mantissa, mantissa_mask));
//
//     // Extract the final result and return it
//     return float_16(static_cast<uint16_t>(_mm_extract_epi16(result_value, 0)));
// }

// Using default libraries of the C++ language
inline float_16 float_16::operator+(const float_16& other) const {
    // Extract components of the first operand
    uint16_t sign1 = value & 0x8000;
    uint16_t exponent1 = (value >> 10) & 0x1F;
    uint16_t mantissa1 = value & 0x03FF;

    // Extract components of the second operand
    const uint16_t sign2 = other.value & 0x8000;
    uint16_t exponent2 = (other.value >> 10) & 0x1F;
    uint16_t mantissa2 = other.value & 0x03FF;

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
        result_mantissa = mantissa1 + mantissa2;
    } else {
        if (mantissa1 >= mantissa2) {
            result_mantissa = mantissa1 - mantissa2;
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


inline float_16 float_16::operator-(const float_16& other) const {
    return float_16(static_cast<float>(*this) - static_cast<float>(other));
}

inline float_16 float_16::operator*(const float_16& other) const {
    return float_16(static_cast<float>(*this) * static_cast<float>(other));
}

inline float_16 float_16::operator/(const float_16& other) const {
    return float_16(static_cast<float>(*this) / static_cast<float>(other));
}

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
    *this += float_16(1.0f);
    return *this;
}

inline float_16 float_16::operator++(int) {
    float_16 temp = *this;
    ++(*this);
    return temp;
}

inline float_16& float_16::operator--() {
    *this -= float_16(1.0f);
    return *this;
}

inline float_16 float_16::operator--(int) {
    float_16 temp = *this;
    --(*this);
    return temp;
}

// Overloaded unary operators
inline float_16 float_16::operator+() const {
    return *this;
}

inline float_16 float_16::operator-() const {
    return float_16(-static_cast<float>(*this));
}

// Overloaded assignment operator
inline float_16& float_16::operator=(const float_16& other) {
    if (this != &other) {
        value = other.value;
    }
    return *this;
}

// Conversion functions
inline uint16_t float_16::float32_to_float16(const float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    const uint16_t sign = (bits >> 16) & 0x8000;
    const uint16_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
    uint16_t mantissa = (bits >> 13) & 0x3ff;

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

// Bit casting function
template <typename To, typename From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value && std::is_trivially_copyable<To>::value, To>::type
bit_cast(const From& src) {
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}


inline float float_16::float16_to_float32(const uint16_t h) {
    const uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7c00) >> 10;
    uint32_t mantissa = (h & 0x03ff) << 13;

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
