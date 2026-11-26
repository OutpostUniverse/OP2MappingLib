// Learned map generator: maps in the style of the original maps (Python:
// mapgen_netz.py, mapgen_netz_waende.py in op2-codegen-editor).
//
// Two small neural networks, trained by fefi-byte on 61 maps (originals and
// OPU), see net.hpp:
//   1. the sketch network lays out walls and grounds on a half-resolution
//      grid, guided by zones (mountain or gorge = tight, plateau or valley =
//      building ground);
//   2. the tile network paints the map tile by tile from the sketch, using
//      only tile pairs that occur in the training maps.
// Mountains, gorges, valleys and plateaus come from mountains.cpp, planned on
// a larger world, so walls run off the map edge.
//
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#pragma once

#include "generate.hpp"
#include "net.hpp"

#include <exception>
#include <functional>

namespace om::gen {

/// The network file: both networks plus the tile tables they need.
struct LearnedModel {
    nn::Weights weights;
    nn::SketchNet sketch;
    nn::TileNet tiles;
    int vocab = 0;                          ///< V: tiles the tile network knows
    std::vector<int32_t> vocabMapping;      ///< vocab index -> standard mapping index
    std::vector<uint8_t> vocabCell;         ///< vocab index -> cell type
    std::vector<uint8_t> vocabWall;         ///< vocab index -> cliff tile?
    std::vector<uint8_t> vocabFree;         ///< vocab index -> passable?
    std::vector<int> wallIdx;               ///< vocab indices of cliff tiles
    /// Pair tables as bit rows: right[a] has bit b if b was seen right of a;
    /// left[b] has bit a. Same for down / up.
    int words = 0;
    std::vector<uint64_t> right, left, down, up;
    /// Per standard mapping index: cell type, ground (-1 none) and flags.
    std::vector<uint8_t> mapCell;
    std::vector<int8_t> mapGround;
    std::vector<uint8_t> mapFlags;          ///< kPlaceable | kTransition | kDetail
    static constexpr uint8_t kPlaceable = 1, kTransition = 2, kDetail = 4;

    /// Throws std::runtime_error for a bad file.
    static std::unique_ptr<LearnedModel> load(const uint8_t* data, size_t len);
    bool seenRight(int a, int b) const { return (right[size_t(a) * words + b / 64] >> (b % 64)) & 1; }
    bool seenDown(int a, int b) const { return (down[size_t(a) * words + b / 64] >> (b % 64)) & 1; }
};

/// Thrown when the progress callback asks to stop.
struct Cancelled : std::exception {
    const char* what() const noexcept override { return "cancelled"; }
};

/// Progress report 0..1; returns false to cancel.
using ProgressFn = std::function<bool(double)>;

/// The sketch of a map: walls and grounds at half resolution, plus the zones.
struct Sketch {
    Grid<int32_t> wall;      ///< (w/2, h/2): 0 none, 1..8 fall direction E, SE, S, SW, W, NW, N, NE
    Grid<int32_t> ground;    ///< (w/2, h/2): 0 mud, 1 rock, 2 sand, 3 vegetation, 4 other (as mud)
    Grid<int8_t> zones;      ///< (w, h): 1 mountain, 2 plateau, 3 valley, 4 gorge
    MountainPlan plan;       ///< the map's cutout of the mountain plan
};

/// Draws the sketch (Python: skizze). Width and height must be multiples of 8.
Sketch drawSketch(const LearnedModel& m, const Params& p, nn::Pool& pool, const ProgressFn& progress);

/// Paints the finished map from the sketch (Python: karte). quality: 1 fast, 2 good.
Result paintMap(const Profile& profile, const LearnedModel& m, const Params& p, const Sketch& sk, int quality,
                nn::Pool& pool, const ProgressFn& progress);

// Shared helpers (learned_sketch.cpp)
/// 8-connected labels from 1 (Python: waende.label), numbered in scan order; returns the count.
int label8(const Mask& m, Grid<int32_t>& labels);
/// Softmax over `n` logits (divided by `temperature`) into `out`, with detExp.
void softmax(const float* logits, int n, double temperature, double* out);

}  // namespace om::gen
