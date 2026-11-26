// The two small neural networks of the learned map generator, with a small
// inference engine of their own (no dependencies).
//
// Results must be the same on every platform and for any thread count, like
// everything else in this library:
//   - every output value is summed in one fixed order, whatever the threads do
//     (threads only split independent outputs between them);
//   - exp and erf are computed here (detmath), not taken from <cmath>;
//   - the build turns off floating-point contraction (no fused multiply-add).
//
// Networks (trained by fefi-byte for op2-codegen-editor, see
// docs/learned-generator.md):
//   SketchNet  walls (fall direction 0..8) and grounds on a half-resolution
//              grid, guided by zones (0 any, 1 tight, 2 building ground);
//   TileNet    the tile of each open cell, given its neighbours and the sketch.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace om::nn {

// ---------------------------------------------------------------------------
// Deterministic math (plain + and *, no library calls besides sqrt and ldexp,
// which are exact).

double detExp(double x);
double detLog(double x);    ///< natural log, x > 0
float detErf(float x);

// ---------------------------------------------------------------------------
// Threads

/// A fixed set of worker threads. run(n, f) calls f(i) for i in [0, n), split
/// between the threads; f must only write outputs that belong to its i.
class Pool {
public:
    explicit Pool(int threads);
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    int threads() const { return static_cast<int>(workers_.size()) + 1; }
    void run(int n, const std::function<void(int)>& f);
    /// While set, run() works on the calling thread only: for small jobs,
    /// where waking the workers costs more than it saves.
    void setSerial(bool serial) { serial_ = serial; }
    bool serial() const { return serial_; }

    /// Makes the pool serial for jobs below `work` units while in scope.
    struct SerialBelow {
        Pool& p;
        bool before;
        SerialBelow(Pool& pool, size_t work, size_t limit) : p(pool), before(pool.serial()) {
            if (work < limit) p.setSerial(true);
        }
        ~SerialBelow() { p.setSerial(before); }
    };

private:
    void work(int id);
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable start_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int n_ = 0, next_ = 0, busy_ = 0;
    uint64_t round_ = 0;
    bool stop_ = false;
    bool serial_ = false;
};

// ---------------------------------------------------------------------------
// Tensors

/// Activations: C x H x W floats, row-major per channel.
struct Act {
    int c = 0, h = 0, w = 0;
    std::vector<float> v;
    Act() = default;
    Act(int channels, int height, int width) : c(channels), h(height), w(width), v(size_t(channels) * height * width, 0.f) {}
    float* ch(int i) { return v.data() + size_t(i) * h * w; }
    const float* ch(int i) const { return v.data() + size_t(i) * h * w; }
    size_t plane() const { return size_t(h) * w; }
};

/// Tensors of the network file, by name (float tensors converted to float32).
class Weights {
public:
    /// Parses the network file. Throws std::runtime_error on bad data.
    static Weights parse(const uint8_t* data, size_t len);
    /// Float tensor with exactly `count` values; throws if missing or of another size.
    const float* f(const std::string& name, size_t count) const;
    const std::vector<uint8_t>& u8(const std::string& name) const;
    const std::vector<uint16_t>& u16(const std::string& name) const;
    std::vector<int> shape(const std::string& name) const;

private:
    struct T {
        std::vector<int> shape;
        std::vector<float> f;
        std::vector<uint8_t> b;
        std::vector<uint16_t> s;
    };
    const T& get(const std::string& name) const;
    std::map<std::string, T> t_;
};

// ---------------------------------------------------------------------------
// Layers

struct Conv {             ///< dense convolution, k x k, stride 1 (same padding) or k (no padding)
    int cin = 0, cout = 0, k = 1, stride = 1;
    const float* w = nullptr;   ///< cout x cin x k x k
    const float* b = nullptr;
    void load(const Weights& wt, const std::string& name, int in, int out, int kernel, int strideK = 1);
    Act operator()(const Act& x, Pool& pool) const;
};

struct Sep {              ///< x + c(gelu(b(dw5(norm(x)))))
    int c = 0;
    const float *ng = nullptr, *nb = nullptr, *aw = nullptr, *ab = nullptr;
    Conv b, cc;
    void load(const Weights& wt, const std::string& name, int channels);
    void operator()(Act& x, Pool& pool) const;
};

/// GroupNorm with 8 groups and affine weights (eps 1e-5), in place.
void groupNorm(Act& x, const float* gamma, const float* beta, Pool& pool);
/// Exact GELU (erf form), in place.
void gelu(Act& x, Pool& pool);
/// Nearest-neighbour upsampling by 2, added onto `into` (into += up(x)).
void addUp2(Act& into, const Act& x);

/// The U-net body both networks share: v1 -> d1 -> h1 -> d2 -> h2, back up
/// through u2/h3 and u1/v2, then GroupNorm.
struct Body {
    int w = 0;
    std::vector<Sep> v1, h1, h2, h3, v2;
    Conv d1, d2, u2, u1;
    const float *ng = nullptr, *nb = nullptr;
    void load(const Weights& wt, const std::string& p, int width, int nh1, int nh2, int nh3, int nv2);
    void operator()(Act& h, Pool& pool) const;
};

/// The sketch network (wmodell). Inputs per cell: wall 0..9 (9 = open),
/// ground 0..5 (5 = open), zone 0..2. Output logits: 9 wall classes, 5 grounds.
struct SketchNet {
    int width = 64;
    const float *ew = nullptr, *eb = nullptr, *ez = nullptr;
    Conv pos, ein, ow, ob;
    Body body;
    void load(const Weights& wt);
    void run(int h, int w, const int32_t* wall, const int32_t* ground, const int32_t* zone,
             Act& wallLogits, Act& groundLogits, Pool& pool) const;
};

/// The tile network (smodell). Inputs per cell: tile 0..V+1 (V = open, V+1 =
/// masked), sketch wall 0..9 and ground 0..5 per cell. Returns the logits of
/// the cells `idx` (row-major indices), V per cell.
struct TileNet {
    int width = 72, vocab = 0;
    const float *emb = nullptr, *ew = nullptr, *eb = nullptr, *bias = nullptr, *ow = nullptr, *obias = nullptr;
    Conv pos, ein, sk0, sk2;
    Body body;
    void load(const Weights& wt);
    /// The sketch part (sk); depends on wall and ground only, so callers may keep it.
    Act sketchFeatures(int h, int w, const int32_t* wall, const int32_t* ground, Pool& pool) const;
    /// The body's output for an h x w window. `pos` (borderPlanes of the whole
    /// map, reach 12) and `sketch` are cut to the window.
    Act features(int h, int w, const int32_t* tiles, const Act& pos, const Act& sketch, Pool& pool) const;
    /// Logits (V per cell) of the cells `idx` (row-major indices into the window).
    void project(const Act& features, const std::vector<int>& idx, std::vector<float>& logits, Pool& pool) const;
    void run(int h, int w, const int32_t* tiles, const Act& pos, const Act& sketch, const std::vector<int>& idx,
             std::vector<float>& logits, Pool& pool) const {
        project(features(h, w, tiles, pos, sketch, pool), idx, logits, pool);
    }
};

/// Border planes (closeness to the four map edges) as the networks expect them.
Act borderPlanes(int h, int w, float reach);

}  // namespace om::nn
