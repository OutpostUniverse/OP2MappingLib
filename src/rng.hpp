// Deterministic random numbers for the cliff generator.
//
// xoshiro256** seeded through splitmix64, plus the few distributions the cliff
// shapes need (uniform, normal, integers, gamma, Dirichlet). Everything is
// specified here rather than taken from <random>, whose distributions differ
// between standard libraries, so every platform and binding draws the same
// numbers for the same seed.
//
// The sequence of calls matters as much as the seed: drawing one more or one
// fewer number anywhere in the cliff generator changes every later number, and
// with it the golden test hashes.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace om {

/// xoshiro256** generator. Copyable; a copy continues the same sequence independently.
class Rng {
public:
    /// Fills the 256-bit state from four splitmix64 outputs, as the xoshiro authors recommend.
    explicit Rng(uint64_t seed) {
        uint64_t z = seed;
        for (auto& s : s_) s = splitmix(z);
    }

    /// Uniform in [0, 1).
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }   // top 53 bits / 2^53

    /// Uniform in [lo, hi).
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

    /// Integer in [lo, hi) (hi exclusive), as numpy's Generator.integers.
    /// Plain modulo, so very slightly biased for large ranges; the ranges used
    /// are tiny. Returns lo for an empty range.
    int64_t integers(int64_t lo, int64_t hi) {
        if (hi <= lo) return lo;
        const uint64_t range = static_cast<uint64_t>(hi - lo);
        return lo + static_cast<int64_t>(next() % range);
    }

    /// Standard normal (Box-Muller, polar form), mean 0, sd 1.
    double normal() {
        // The polar method makes two values per round; the second is kept for the next call.
        if (hasSpare_) {
            hasSpare_ = false;
            return spare_;
        }
        double u, v, s;
        do {
            u = uniform(-1.0, 1.0);
            v = uniform(-1.0, 1.0);
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double f = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = v * f;
        hasSpare_ = true;
        return u * f;
    }

    /// Gamma(shape, 1) by Marsaglia and Tsang (shape > 0).
    double gamma(double shape) {
        // shape < 1: boost to shape + 1 and scale by u^(1/shape) (Marsaglia and Tsang).
        if (shape < 1.0) {
            const double u = uniform();
            return gamma(shape + 1.0) * std::pow(u > 0 ? u : 1e-300, 1.0 / shape);
        }
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do {
                x = normal();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            const double u = uniform();
            if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
            if (std::log(u > 0 ? u : 1e-300) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
        }
    }

    /// Dirichlet sample with k equal parameters alpha (sums to 1).
    std::vector<double> dirichlet(int k, double alpha) {
        std::vector<double> out(static_cast<size_t>(k));
        double sum = 0;
        for (double& v : out) {
            v = gamma(alpha);
            sum += v;
        }
        for (double& v : out) v = sum > 0 ? v / sum : 1.0 / k;
        return out;
    }

private:
    /// One splitmix64 step: advances x and returns the mixed value.
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    /// Next raw 64-bit output (xoshiro256** reference algorithm).
    uint64_t next() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    uint64_t s_[4];            // generator state
    bool hasSpare_ = false;
    double spare_ = 0;
};

}  // namespace om
