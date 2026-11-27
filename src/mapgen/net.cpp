// Inference engine of the learned map generator; see net.hpp.
#include "net.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86_FP))) || defined(OM_FORCE_SSE)
#include <emmintrin.h>
#endif

namespace om::nn {

// ---------------------------------------------------------------------------
// Deterministic math

double detExp(double x) {
    if (x < -745.0) return 0.0;
    if (x > 709.0) return HUGE_VAL;
    const double ln2hi = 6.93147180369123816490e-01, ln2lo = 1.90821492927058770002e-10;
    const double k = std::floor(x * 1.44269504088896338700 + 0.5);
    const double r = (x - k * ln2hi) - k * ln2lo;   // |r| <= 0.35
    // Taylor series up to r^13 / 13!: the rest is below 1e-17 for |r| <= 0.35.
    double p = 1.0 / 6227020800.0;
    static const double c[13] = {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0,
                                 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0};
    for (double ci : c) p = p * r + ci;
    return std::ldexp(p, static_cast<int>(k));
}

double detLog(double x) {
    if (!(x > 0)) return -HUGE_VAL;
    int e = 0;
    double m = std::frexp(x, &e);   // x = m * 2^e, m in [0.5, 1)
    if (m < 0.70710678118654752440) {
        m *= 2.0;
        --e;
    }
    // log(m) = 2 atanh(f), f = (m - 1) / (m + 1), |f| <= 0.172
    const double f = (m - 1.0) / (m + 1.0), s = f * f;
    double p = 0.0;
    for (int n = 23; n >= 1; n -= 2) p = p * s + 1.0 / n;
    return e * 0.69314718055994530942 + 2.0 * f * p;
}

float detErf(float xf) {
    // Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
    const double x = xf, ax = std::fabs(x);
    const double t = 1.0 / (1.0 + 0.3275911 * ax);
    const double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
                               detExp(-ax * ax);
    return static_cast<float>(x < 0 ? -y : y);
}

// ---------------------------------------------------------------------------
// Pool

Pool::Pool(int threads) {
    for (int i = 1; i < std::max(1, threads); ++i) workers_.emplace_back([this, i] { work(i); });
}

Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    start_.notify_all();
    for (auto& t : workers_) t.join();
}

void Pool::work(int) {
    uint64_t seen = 0;
    for (;;) {
        const std::function<void(int)>* job;
        {
            std::unique_lock<std::mutex> lk(m_);
            start_.wait(lk, [&] { return stop_ || round_ != seen; });
            if (stop_) return;
            seen = round_;
            job = job_;
            ++busy_;
        }
        for (;;) {
            int i;
            {
                // Only items of the round this job belongs to: the caller may
                // already have finished it and started the next one.
                std::lock_guard<std::mutex> lk(m_);
                if (round_ != seen || !job || next_ >= n_) break;
                i = next_++;
            }
            (*job)(i);
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            --busy_;
        }
        done_.notify_all();
    }
}

void Pool::run(int n, const std::function<void(int)>& f) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1 || serial_) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(m_);
        job_ = &f;
        n_ = n;
        next_ = 0;
        ++round_;
    }
    start_.notify_all();
    for (;;) {
        int i;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (next_ >= n_) break;
            i = next_++;
        }
        f(i);
    }
    std::unique_lock<std::mutex> lk(m_);
    done_.wait(lk, [&] { return busy_ == 0 && next_ >= n_; });
    job_ = nullptr;
}

// ---------------------------------------------------------------------------
// Weights

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    void need(size_t n) const {
        if (static_cast<size_t>(end - p) < n) throw std::runtime_error("network file is truncated");
    }
    uint32_t u32() {
        need(4);
        const uint32_t v = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
        p += 4;
        return v;
    }
    uint16_t u16() {
        need(2);
        const uint16_t v = static_cast<uint16_t>(p[0] | p[1] << 8);
        p += 2;
        return v;
    }
    uint8_t u8() {
        need(1);
        return *p++;
    }
};

float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {   // subnormal: normalise
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                --exp;
            }
            bits = sign | exp << 23 | (man & 0x3ff) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | man << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | man << 13;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

Weights Weights::parse(const uint8_t* data, size_t len) {
    if (!data || len < 12 || std::memcmp(data, "OMNN", 4) != 0) throw std::runtime_error("not a network file (OMNN)");
    Reader r{data + 4, data + len};
    if (r.u32() != 1) throw std::runtime_error("unsupported network file version");
    const uint32_t count = r.u32();
    Weights out;
    for (uint32_t i = 0; i < count; ++i) {
        const uint16_t nl = r.u16();
        r.need(nl);
        std::string name(reinterpret_cast<const char*>(r.p), nl);
        r.p += nl;
        const uint8_t type = r.u8(), rank = r.u8();
        T t;
        size_t n = 1;
        for (int d = 0; d < rank; ++d) {
            t.shape.push_back(static_cast<int>(r.u32()));
            n *= static_cast<size_t>(t.shape.back());
        }
        switch (type) {
        case 0:
            r.need(n * 4);
            t.f.resize(n);
            for (size_t k = 0; k < n; ++k) {
                const uint32_t bits = r.u32();
                std::memcpy(&t.f[k], &bits, 4);
            }
            break;
        case 1:
            r.need(n * 2);
            t.f.resize(n);
            for (size_t k = 0; k < n; ++k) t.f[k] = halfToFloat(r.u16());
            break;
        case 2:
            r.need(n);
            t.b.assign(r.p, r.p + n);
            r.p += n;
            break;
        case 3:
            r.need(n * 2);
            t.s.resize(n);
            for (size_t k = 0; k < n; ++k) t.s[k] = r.u16();
            break;
        default:
            throw std::runtime_error("network file: unknown tensor type in " + name);
        }
        out.t_[name] = std::move(t);
    }
    return out;
}

const Weights::T& Weights::get(const std::string& name) const {
    auto it = t_.find(name);
    if (it == t_.end()) throw std::runtime_error("network file lacks tensor " + name);
    return it->second;
}

const float* Weights::f(const std::string& name, size_t count) const {
    const T& t = get(name);
    if (t.f.size() != count) throw std::runtime_error("network file: tensor " + name + " has the wrong size");
    return t.f.data();
}
const std::vector<uint8_t>& Weights::u8(const std::string& name) const { return get(name).b; }
const std::vector<uint16_t>& Weights::u16(const std::string& name) const { return get(name).s; }
std::vector<int> Weights::shape(const std::string& name) const { return get(name).shape; }

// ---------------------------------------------------------------------------
// Layers

namespace {

// Hot loops get an AVX2 variant where the toolchain can pick one at run time.
// No FMA: a fused multiply-add rounds differently, and every platform must
// give the same bits (the build also disables contraction).
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)) && defined(__linux__) && !defined(__clang__)
#define OM_HOT __attribute__((target_clones("avx2", "default")))
#else
#define OM_HOT
#endif

constexpr int kCoTask = 32;    // output channels per task (a multiple of kG)
constexpr int kPTask = 512;    // pixels per task
constexpr int kR = 4;   // convolution register block: 4 output channels x 16 pixels

// 8-float vectors: GCC/Clang vector types, SSE2 intrinsics for MSVC on x86.
// Without either the scalar loops run; every path does the same operations
// per value in the same order (and no fused multiply-add), so all give the
// same bits.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(OM_NO_VEC) && !defined(OM_FORCE_SSE)
#define OM_VEC 1
#pragma GCC diagnostic ignored "-Wpsabi"   // 32-byte vectors without AVX: internal functions only
typedef float v8 __attribute__((vector_size(32)));
typedef int32_t vi8 __attribute__((vector_size(32)));
inline v8 load8(const float* p) {
    v8 v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline void store8(float* p, v8 v) { std::memcpy(p, &v, sizeof v); }
inline v8 splat(float f) { return v8{f, f, f, f, f, f, f, f}; }
inline v8 vabs(v8 x) { return x < 0 ? -x : x; }
inline v8 vsignOf(v8 x, v8 y) { return x < 0 ? -y : y; }      // -y where x < 0
inline v8 vatLeast(v8 z, float lo) { return z < splat(lo) ? splat(lo) : z; }
inline vi8 vtrunc(v8 x) { return __builtin_convertvector(x, vi8); }
inline v8 vfloat(vi8 n) { return __builtin_convertvector(n, v8); }
inline v8 vpow2(vi8 n) {   // 2^n for n in the normal range
    const vi8 bits = (n + 127) << 23;
    v8 v;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}
#elif ((defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86_FP))) || defined(OM_FORCE_SSE)) && !defined(OM_NO_VEC)
#define OM_VEC 1
struct v8 { __m128 a, b; };
struct vi8 { __m128i a, b; };
inline v8 operator+(v8 x, v8 y) { return {_mm_add_ps(x.a, y.a), _mm_add_ps(x.b, y.b)}; }
inline v8 operator-(v8 x, v8 y) { return {_mm_sub_ps(x.a, y.a), _mm_sub_ps(x.b, y.b)}; }
inline v8 operator*(v8 x, v8 y) { return {_mm_mul_ps(x.a, y.a), _mm_mul_ps(x.b, y.b)}; }
inline v8 operator/(v8 x, v8 y) { return {_mm_div_ps(x.a, y.a), _mm_div_ps(x.b, y.b)}; }
inline v8 operator-(v8 x) { return {_mm_xor_ps(x.a, _mm_set1_ps(-0.f)), _mm_xor_ps(x.b, _mm_set1_ps(-0.f))}; }
inline v8& operator+=(v8& x, v8 y) { return x = x + y; }
inline v8 load8(const float* p) { return {_mm_loadu_ps(p), _mm_loadu_ps(p + 4)}; }
inline void store8(float* p, v8 v) {
    _mm_storeu_ps(p, v.a);
    _mm_storeu_ps(p + 4, v.b);
}
inline v8 splat(float f) { return {_mm_set1_ps(f), _mm_set1_ps(f)}; }
inline __m128 sel(__m128 m, __m128 yes, __m128 no) { return _mm_or_ps(_mm_and_ps(m, yes), _mm_andnot_ps(m, no)); }
inline v8 vsignOf(v8 x, v8 y) {
    const v8 n = -y;
    const __m128 z = _mm_setzero_ps();
    return {sel(_mm_cmplt_ps(x.a, z), n.a, y.a), sel(_mm_cmplt_ps(x.b, z), n.b, y.b)};
}
inline v8 vabs(v8 x) { return vsignOf(x, x); }
inline v8 vatLeast(v8 z, float lo) {
    const __m128 l = _mm_set1_ps(lo);
    return {sel(_mm_cmplt_ps(z.a, l), l, z.a), sel(_mm_cmplt_ps(z.b, l), l, z.b)};
}
inline vi8 vtrunc(v8 x) { return {_mm_cvttps_epi32(x.a), _mm_cvttps_epi32(x.b)}; }
inline v8 vfloat(vi8 n) { return {_mm_cvtepi32_ps(n.a), _mm_cvtepi32_ps(n.b)}; }
inline v8 vpow2(vi8 n) {
    const __m128i k = _mm_set1_epi32(127);
    return {_mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n.a, k), 23)),
            _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n.b, k), 23))};
}
#endif

constexpr int kG = 4;   // gemm register block: 4 output channels x 16 pixels

/// Register blocks of gemm: rows [co0, co1) x pixels [p0, p1). Wp holds the
/// weights of each block of kG rows interleaved: Wp[(blk * jn + j) * kG + r].
OM_HOT void gemmBlock(int co0, int co1, int p0, int p1, int jn, const float* W, const float* Wp, const float* b,
                      const float* const* rows, int P, float* out) {
    const int coFull = co0 + (co1 - co0) / kG * kG;
#ifdef OM_VEC
    int ps = p0;
    // A 16-pixel strip of every input row stays in L1 while all channel blocks use it.
    for (; ps + 16 <= p1; ps += 16)
        for (int cs = co0; cs < coFull; cs += kG) {
            v8 a[kG][2];
            for (int r = 0; r < kG; ++r) a[r][0] = a[r][1] = splat(b ? b[cs + r] : 0.f);
            const float* w0 = Wp + size_t(cs / kG) * jn * kG;
            for (int j = 0; j < jn; ++j) {
                const v8 x0 = load8(rows[j] + ps), x1 = load8(rows[j] + ps + 8);
                const float* wj = w0 + size_t(j) * kG;
                for (int r = 0; r < kG; ++r) {
                    const v8 wv = splat(wj[r]);
                    a[r][0] += wv * x0;
                    a[r][1] += wv * x1;
                }
            }
            for (int r = 0; r < kG; ++r) {
                store8(out + size_t(cs + r) * P + ps, a[r][0]);
                store8(out + size_t(cs + r) * P + ps + 8, a[r][1]);
            }
        }
    const int vecEnd = ps;
#else
    (void)Wp;
    const int vecEnd = p0;
#endif
    // Everything the vector loop left: the pixel tail, and channels past the last full block.
    for (int co = co0; co < co1; ++co)
        for (int p = (co < coFull ? vecEnd : p0); p < p1; ++p) {
            float acc = b ? b[co] : 0.f;
            for (int j = 0; j < jn; ++j) acc += W[size_t(co) * jn + j] * rows[j][p];
            out[size_t(co) * P + p] = acc;
        }
}

/// One output row of a k x k convolution over a zero-padded input, channels [cs, cs + kR).
OM_HOT void convRow(int task, int H, int Wd, int cin, int cout, int k, int ph, int pw, const float* w, const float* b,
                    const float* xp, Act& y) {
    const int cs = (task / H) * kR, yy = task % H, nr = std::min(kR, cout - cs);
    int xs = 0;
#ifdef OM_VEC
    if (nr == kR)
        for (; xs + 16 <= Wd; xs += 16) {
            v8 a[kR][2];
            for (int r = 0; r < kR; ++r) a[r][0] = a[r][1] = splat(b[cs + r]);
            for (int c = 0; c < cin; ++c)
                for (int ky = 0; ky < k; ++ky) {
                    const float* row = xp + (size_t(c) * ph + yy + ky) * pw + xs;
                    for (int kx = 0; kx < k; ++kx) {
                        const v8 x0 = load8(row + kx), x1 = load8(row + kx + 8);
                        for (int r = 0; r < kR; ++r) {
                            const v8 wv = splat(w[((size_t(cs + r) * cin + c) * k + ky) * k + kx]);
                            a[r][0] += wv * x0;
                            a[r][1] += wv * x1;
                        }
                    }
                }
            for (int r = 0; r < kR; ++r) {
                store8(y.ch(cs + r) + size_t(yy) * Wd + xs, a[r][0]);
                store8(y.ch(cs + r) + size_t(yy) * Wd + xs + 8, a[r][1]);
            }
        }
#endif
    for (; xs < Wd; ++xs)
        for (int r = 0; r < nr; ++r) {
            float acc = b[cs + r];
            for (int c = 0; c < cin; ++c)
                for (int ky = 0; ky < k; ++ky)
                    for (int kx = 0; kx < k; ++kx)
                        acc += w[((size_t(cs + r) * cin + c) * k + ky) * k + kx] * xp[(size_t(c) * ph + yy + ky) * pw + xs + kx];
            y.ch(cs + r)[size_t(yy) * Wd + xs] = acc;
        }
}

/// Depthwise 5x5 of one channel over its zero-padded plane; terms in (ky, kx) order.
OM_HOT void dwChannel(const float* s, float* o, const float* k, float bias, int H, int W, int pw) {
    for (int y = 0; y < H; ++y) {
        int xs = 0;
#ifdef OM_VEC
        for (; xs + 8 <= W; xs += 8) {
            v8 a = splat(bias);
            for (int ky = 0; ky < 5; ++ky) {
                const float* row = s + size_t(y + ky) * pw + xs;
                for (int kx = 0; kx < 5; ++kx) a += splat(k[ky * 5 + kx]) * load8(row + kx);
            }
            store8(o + size_t(y) * W + xs, a);
        }
#endif
        for (; xs < W; ++xs) {
            float a = bias;
            for (int ky = 0; ky < 5; ++ky)
                for (int kx = 0; kx < 5; ++kx) a += k[ky * 5 + kx] * s[size_t(y + ky) * pw + xs + kx];
            o[size_t(y) * W + xs] = a;
        }
    }
}

/// out[co][p] = b[co] + sum_j W[co][j] * rows[j][p], j in order (whatever the blocking).
void gemm(int cout, int jn, const float* W, const float* b, const std::vector<const float*>& rows, int P, float* out,
          Pool& pool) {
    // Tasks of kCoTask channels (a multiple of kG) x kPTask pixels.
    const int cb = (cout + kCoTask - 1) / kCoTask, pb = (P + kPTask - 1) / kPTask;
    std::vector<float> Wp(size_t(cout / kG) * jn * kG);
    for (int blk = 0; blk < cout / kG; ++blk)
        for (int j = 0; j < jn; ++j)
            for (int r = 0; r < kG; ++r) Wp[(size_t(blk) * jn + j) * kG + r] = W[size_t(blk * kG + r) * jn + j];
    pool.run(cb * pb, [&](int task) {
        const int co0 = (task / pb) * kCoTask, p0 = (task % pb) * kPTask;
        const int co1 = std::min(cout, co0 + kCoTask), p1 = std::min(P, p0 + kPTask);
        gemmBlock(co0, co1, p0, p1, jn, W, Wp.data(), b, rows.data(), P, out);
    });
}

/// Zero-padded copy of x (pad cells on every side).
std::vector<float> padded(const Act& x, int pad, Pool& pool) {
    const int ph = x.h + 2 * pad, pw = x.w + 2 * pad;
    std::vector<float> o(size_t(x.c) * ph * pw, 0.f);
    pool.run(x.c, [&](int c) {
        for (int y = 0; y < x.h; ++y)
            std::memcpy(&o[(size_t(c) * ph + y + pad) * pw + pad], x.ch(c) + size_t(y) * x.w, sizeof(float) * size_t(x.w));
    });
    return o;
}

}  // namespace

void Conv::load(const Weights& wt, const std::string& name, int in, int out, int kernel, int strideK) {
    cin = in;
    cout = out;
    k = kernel;
    stride = strideK;
    w = wt.f(name + ".weight", size_t(out) * in * kernel * kernel);
    b = wt.f(name + ".bias", size_t(out));
}

Act Conv::operator()(const Act& x, Pool& pool) const {
    if (x.c != cin) throw std::runtime_error("conv: channel count mismatch");
    if (k == 1) {
        Act y(cout, x.h, x.w);
        std::vector<const float*> rows(static_cast<size_t>(cin));
        for (int c = 0; c < cin; ++c) rows[size_t(c)] = x.ch(c);
        gemm(cout, cin, w, b, rows, int(x.plane()), y.v.data(), pool);
        return y;
    }
    if (stride == k) {   // k x k, stride k, no padding: rearrange, then a 1x1 product
        const int oh = x.h / k, ow = x.w / k, P = oh * ow;
        std::vector<float> col(size_t(cin) * k * k * P);
        std::vector<const float*> rows(size_t(cin) * k * k);
        for (int c = 0; c < cin; ++c)
            for (int ky = 0; ky < k; ++ky)
                for (int kx = 0; kx < k; ++kx) {
                    const size_t j = (size_t(c) * k + ky) * k + kx;
                    float* d = col.data() + j * P;
                    const float* s = x.ch(c);
                    for (int yy = 0; yy < oh; ++yy)
                        for (int xx = 0; xx < ow; ++xx) d[yy * ow + xx] = s[size_t(yy * k + ky) * x.w + xx * k + kx];
                    rows[j] = d;
                }
        Act y(cout, oh, ow);
        gemm(cout, cin * k * k, w, b, rows, P, y.v.data(), pool);
        return y;
    }
    // k x k, stride 1, same padding: over a zero-padded copy, terms in (c, ky, kx) order
    const int pad = k / 2, H = x.h, Wd = x.w, pw = Wd + 2 * pad, ph = H + 2 * pad;
    const std::vector<float> xp = padded(x, pad, pool);
    Act y(cout, H, Wd);
    const int cb = (cout + kR - 1) / kR;
    pool.run(cb * H, [&](int task) {
        convRow(task, H, Wd, cin, cout, k, ph, pw, w, b, xp.data(), y);
    });
    return y;
}

namespace {

/// Sum of n floats in a fixed order: 8 lanes (element i goes to lane i % 8),
/// then the lanes in order. Vectorises, and gives the same bits everywhere.
OM_HOT double laneSum(const float* v, size_t n, double centre) {
    double lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; ++l) lane[l] += double(v[i + size_t(l)]) - centre;
    for (; i < n; ++i) lane[i % 8] += double(v[i]) - centre;
    double s = 0;
    for (double l : lane) s += l;
    return s;
}

OM_HOT double laneSquares(const float* v, size_t n, double mean) {
    double lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; ++l) {
            const double d = double(v[i + size_t(l)]) - mean;
            lane[l] += d * d;
        }
    for (; i < n; ++i) {
        const double d = double(v[i]) - mean;
        lane[i % 8] += d * d;
    }
    double s = 0;
    for (double l : lane) s += l;
    return s;
}

}  // namespace

void groupNorm(Act& x, const float* gamma, const float* beta, Pool& pool) {
    const int groups = 8, per = x.c / groups;
    const size_t P = x.plane();
    // Per channel in parallel, then per group in channel order.
    std::vector<double> part(static_cast<size_t>(x.c));
    pool.run(x.c, [&](int c) { part[size_t(c)] = laneSum(x.ch(c), P, 0.0); });
    std::vector<double> mean(groups), inv(groups);
    for (int g = 0; g < groups; ++g) {
        double s = 0;
        for (int c = g * per; c < (g + 1) * per; ++c) s += part[size_t(c)];
        mean[size_t(g)] = s / (double(per) * double(P));
    }
    pool.run(x.c, [&](int c) { part[size_t(c)] = laneSquares(x.ch(c), P, mean[size_t(c / per)]); });
    for (int g = 0; g < groups; ++g) {
        double s = 0;
        for (int c = g * per; c < (g + 1) * per; ++c) s += part[size_t(c)];
        inv[size_t(g)] = 1.0 / std::sqrt(s / (double(per) * double(P)) + 1e-5);
    }
    pool.run(x.c, [&](int c) {
        float* s = x.ch(c);
        const int g = c / per;
        const float m = static_cast<float>(mean[size_t(g)]), sc = static_cast<float>(inv[size_t(g)]) * gamma[c], sh = beta[c];
        for (size_t p = 0; p < P; ++p) s[p] = (s[p] - m) * sc + sh;
    });
}

namespace {

// GELU with erf from Abramowitz and Stegun 7.1.26 and a float exp of its own:
// the vector and the scalar path do the same float operations in the same
// order, so every build gives the same bits.
constexpr float kLog2e = 1.44269504f, kLn2Hi = 0.693145752f, kLn2Lo = 1.42860677e-6f;
constexpr float kE2 = 1.f / 2, kE3 = 1.f / 6, kE4 = 1.f / 24, kE5 = 1.f / 120, kE6 = 1.f / 720, kE7 = 1.f / 5040;

/// exp(z) for z in [-87, 0].
inline float expNeg(float z) {
    const int n = static_cast<int>(z * kLog2e - 0.5f);
    const float nf = static_cast<float>(n);
    const float r = (z - nf * kLn2Hi) - nf * kLn2Lo;
    const float p = ((((((kE7 * r + kE6) * r + kE5) * r + kE4) * r + kE3) * r + kE2) * r + 1.f) * r + 1.f;
    const int32_t bits = (n + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, 4);
    return p * scale;
}

inline float geluOne(float x) {
    const float a = std::fabs(x) * 0.707106781f;
    const float t = 1.f / (1.f + 0.3275911f * a);
    const float z = -(a * a);
    const float e = expNeg(z < -87.f ? -87.f : z);
    const float y = 1.f - ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * e;
    const float erf = x < 0 ? -y : y;
    return 0.5f * x * (1.f + erf);
}

#ifdef OM_VEC
OM_HOT void geluRow(float* s, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const v8 x = load8(s + i);
        const v8 a = vabs(x) * splat(0.707106781f);
        const v8 t = splat(1.f) / (splat(1.f) + splat(0.3275911f) * a);
        const v8 z = vatLeast(-(a * a), -87.f);
        const vi8 n8 = vtrunc(z * splat(kLog2e) - splat(0.5f));
        const v8 nf = vfloat(n8);
        const v8 r = (z - nf * splat(kLn2Hi)) - nf * splat(kLn2Lo);
        const v8 p = ((((((splat(kE7) * r + splat(kE6)) * r + splat(kE5)) * r + splat(kE4)) * r + splat(kE3)) * r + splat(kE2)) * r +
                      splat(1.f)) * r + splat(1.f);
        const v8 e = p * vpow2(n8);
        const v8 y = splat(1.f) - ((((splat(1.061405429f) * t - splat(1.453152027f)) * t + splat(1.421413741f)) * t -
                                   splat(0.284496736f)) * t + splat(0.254829592f)) * t * e;
        store8(s + i, splat(0.5f) * x * (splat(1.f) + vsignOf(x, y)));
    }
    for (; i < n; ++i) s[i] = geluOne(s[i]);
}
#else
void geluRow(float* s, size_t n) {
    for (size_t i = 0; i < n; ++i) s[i] = geluOne(s[i]);
}
#endif

}  // namespace

void gelu(Act& x, Pool& pool) {
    pool.run(x.c, [&](int c) { geluRow(x.ch(c), x.plane()); });
}

void addUp2(Act& into, const Act& x) {
    for (int c = 0; c < into.c; ++c) {
        float* d = into.ch(c);
        const float* s = x.ch(c);
        for (int y = 0; y < into.h; ++y)
            for (int xx = 0; xx < into.w; ++xx) d[size_t(y) * into.w + xx] += s[size_t(y / 2) * x.w + xx / 2];
    }
}

void Sep::load(const Weights& wt, const std::string& name, int channels) {
    c = channels;
    ng = wt.f(name + ".n.weight", size_t(c));
    nb = wt.f(name + ".n.bias", size_t(c));
    aw = wt.f(name + ".a.weight", size_t(c) * 25);
    ab = wt.f(name + ".a.bias", size_t(c));
    b.load(wt, name + ".b", c, 2 * c, 1);
    cc.load(wt, name + ".c", 2 * c, c, 1);
}

void Sep::operator()(Act& x, Pool& pool) const {
    Act t = x;
    groupNorm(t, ng, nb, pool);
    Act d(c, x.h, x.w);
    const int H = x.h, W = x.w;
    const int pw = W + 4, ph = H + 4;
    const std::vector<float> tp = padded(t, 2, pool);
    pool.run(c, [&](int ch) {   // depthwise 5x5, padding 2, terms in (ky, kx) order
        dwChannel(&tp[size_t(ch) * ph * pw], d.ch(ch), aw + size_t(ch) * 25, ab[ch], H, W, pw);
    });
    Act e = b(d, pool);
    gelu(e, pool);
    const Act f = cc(e, pool);
    for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += f.v[i];
}

void Body::load(const Weights& wt, const std::string& p, int width, int nh1, int nh2, int nh3, int nv2) {
    w = width;
    auto seps = [&](std::vector<Sep>& out, const std::string& name, int n, int ch) {
        out.resize(size_t(n));
        for (int i = 0; i < n; ++i) out[size_t(i)].load(wt, p + name + "." + std::to_string(i), ch);
    };
    seps(v1, "v1", 2, w);
    d1.load(wt, p + "d1", w, 2 * w, 2, 2);
    seps(h1, "h1", nh1, 2 * w);
    d2.load(wt, p + "d2", 2 * w, 2 * w, 2, 2);
    seps(h2, "h2", nh2, 2 * w);
    u2.load(wt, p + "u2", 2 * w, 2 * w, 1);
    seps(h3, "h3", nh3, 2 * w);
    u1.load(wt, p + "u1", 2 * w, w, 1);
    seps(v2, "v2", nv2, w);
    ng = wt.f(p + "n.weight", size_t(w));
    nb = wt.f(p + "n.bias", size_t(w));
}

void Body::operator()(Act& h, Pool& pool) const {
    for (const Sep& s : v1) s(h, pool);
    Act a = d1(h, pool);
    for (const Sep& s : h1) s(a, pool);
    Act c = d2(a, pool);
    for (const Sep& s : h2) s(c, pool);
    addUp2(a, u2(c, pool));
    for (const Sep& s : h3) s(a, pool);
    addUp2(h, u1(a, pool));
    for (const Sep& s : v2) s(h, pool);
    groupNorm(h, ng, nb, pool);
}

Act borderPlanes(int h, int w, float reach) {
    Act p(4, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            auto f = [&](float d) { return std::min(1.f, std::max(0.f, 1.f - d / reach)); };
            p.ch(0)[i] = f(float(y));
            p.ch(1)[i] = f(float(h - 1 - y));
            p.ch(2)[i] = f(float(x));
            p.ch(3)[i] = f(float(w - 1 - x));
        }
    return p;
}

void SketchNet::load(const Weights& wt) {
    const std::string p = "sketch.";
    width = wt.shape(p + "ew.weight").at(1);
    ew = wt.f(p + "ew.weight", size_t(10) * width);
    eb = wt.f(p + "eb.weight", size_t(6) * width);
    ez = wt.f(p + "ez.weight", size_t(3) * width);
    pos.load(wt, p + "p", 4, width, 1);
    ein.load(wt, p + "ein", width, width, 3);
    body.load(wt, p, width, 2, 3, 2, 2);
    ow.load(wt, p + "ow", width, 9, 1);
    ob.load(wt, p + "ob", width, 5, 1);
}

void SketchNet::run(int h, int w, const int32_t* wall, const int32_t* ground, const int32_t* zone, Act& wallLogits,
                    Act& groundLogits, Pool& pool) const {
    Act x = pos(borderPlanes(h, w, 8.f), pool);
    const size_t P = size_t(h) * w;
    for (int c = 0; c < width; ++c) {
        float* d = x.ch(c);
        for (size_t i = 0; i < P; ++i)
            d[i] = ((ew[size_t(wall[i]) * width + c] + eb[size_t(ground[i]) * width + c]) + ez[size_t(zone[i]) * width + c]) + d[i];
    }
    Act hh = ein(x, pool);
    body(hh, pool);
    wallLogits = ow(hh, pool);
    groundLogits = ob(hh, pool);
}

void TileNet::load(const Weights& wt) {
    const std::string p = "tiles.";
    const auto es = wt.shape(p + "emb.weight");
    width = es.at(1);
    vocab = es.at(0) - 2;
    emb = wt.f(p + "emb.weight", size_t(vocab + 2) * width);
    ew = wt.f(p + "ew.weight", size_t(10) * width);
    eb = wt.f(p + "eb.weight", size_t(6) * width);
    bias = wt.f(p + "bias", size_t(vocab));
    ow = wt.f(p + "o.weight", size_t(width) * width);
    obias = wt.f(p + "o.bias", size_t(width));
    pos.load(wt, p + "pos", 4, width, 1);
    ein.load(wt, p + "ein", width, width, 3);
    sk0.load(wt, p + "sk.0", width, width, 5);
    sk2.load(wt, p + "sk.2", width, width, 1);
    body.load(wt, p, width, 3, 3, 2, 3);
}

Act TileNet::sketchFeatures(int h, int w, const int32_t* wall, const int32_t* ground, Pool& pool) const {
    Pool::SerialBelow small(pool, size_t(h) * w, 256);
    Act s(width, h, w);
    const size_t P = size_t(h) * w;
    for (int c = 0; c < width; ++c) {
        float* d = s.ch(c);
        for (size_t i = 0; i < P; ++i) d[i] = ew[size_t(wall[i]) * width + c] + eb[size_t(ground[i]) * width + c];
    }
    Act k = sk0(s, pool);
    gelu(k, pool);
    return sk2(k, pool);
}

Act TileNet::features(int h, int w, const int32_t* tiles, const Act& posPlanes, const Act& sketch, Pool& pool) const {
    Pool::SerialBelow small(pool, size_t(h) * w, 256);
    Act x = pos(posPlanes, pool);
    const size_t P = size_t(h) * w;
    for (int c = 0; c < width; ++c) {
        float* d = x.ch(c);
        const float* k = sketch.ch(c);
        for (size_t i = 0; i < P; ++i) d[i] = (emb[size_t(tiles[i]) * width + c] + d[i]) + k[i];
    }
    Act hh = ein(x, pool);
    body(hh, pool);
    return hh;
}

void TileNet::project(const Act& hh, const std::vector<int>& idx, std::vector<float>& logits, Pool& pool) const {
    Pool::SerialBelow small(pool, idx.size(), 64);
    const int n = static_cast<int>(idx.size()), V = vocab, C = width;
    logits.assign(size_t(n) * V, 0.f);
    const int blocks = (n + 15) / 16;
    pool.run(blocks, [&](int bi) {
        std::vector<float> hv(static_cast<size_t>(C)), o(static_cast<size_t>(C));
        for (int i = bi * 16; i < std::min(n, bi * 16 + 16); ++i) {
            for (int c = 0; c < C; ++c) hv[size_t(c)] = hh.ch(c)[size_t(idx[size_t(i)])];
            for (int r = 0; r < C; ++r) {
                float a = obias[r];
                for (int c = 0; c < C; ++c) a += ow[size_t(r) * C + c] * hv[size_t(c)];
                o[size_t(r)] = a;
            }
            float* out = logits.data() + size_t(i) * V;
            for (int j = 0; j < V; ++j) {
                const float* e = emb + size_t(j) * C;
                float a = 0.f;
                for (int c = 0; c < C; ++c) a += o[size_t(c)] * e[c];
                out[j] = a / 8.f + bias[j];
            }
        }
    });
}

}  // namespace om::nn
