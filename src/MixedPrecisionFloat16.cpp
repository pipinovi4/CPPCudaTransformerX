// Last modified on: 03/07/2024

#include "../include/MixedPrecisionFloat16.h"

float_16::float_16() : value(0) {};

float_16::float_16(float f) {
    value = float32_to_float16(f);
}

float_16::operator float() const {
    return float16_to_float32(value);
}

uint16_t float_16::float32_to_float16(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (bits >> 16) & 0x8000;
    uint16_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
    uint16_t mantissa = (bits >> 13) & 0x3ff;

    if (exponent <= 0) {
        return sign;
    } else if (exponent > 30) {
        return sign | 0x7c00 | (mantissa == 0 ? 0 : 1);
    } else {
        return sign | (exponent << 10) | mantissa;
    }
}

float float_16::float16_to_float32(uint16_t h) {
    uint16_t sign = (h >> 15) & 0x1;
    uint16_t exponent = (h >> 10) & 0x1f;
    uint16_t mantissa = h & 0x3ff;

    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign << 31;
        } else {
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent++;
            }
            mantissa &= ~0x400;
            result = (sign << 31) | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        result = (sign << 31) | 0x7f800000 | (mantissa << 13);
    } else {
        result = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return *reinterpret_cast<float*>(&result);
}

float_16 operator+(const float_16& a, const float_16& b) {
    uint16_t result = a.value + b.value;
    return float_16(result);
}

float_16 operator-(const float_16& a, const float_16& b) {
    uint16_t result = a.value - b.value;
    return float_16(result);
}

float_16 operator*(const float_16& a, const float_16& b) {
    uint16_t result = a.value * b.value;
    return float_16(result);
}

float_16 operator/(const float_16& a, const float_16& b) {
    uint16_t result = a.value / b.value;
    return float_16(result);
}

