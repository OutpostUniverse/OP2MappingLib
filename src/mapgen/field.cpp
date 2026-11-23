// Map generator - biome field.
//
// Port of op2-codegen-editor's mapview/mapgen_felder.py (noise and family
// fields, smoothing, symmetry, volcano sites, lava zones), _bewuchs_abstand
// from mapgen_uebergaenge.py and the TEMPLATES table from mapgen_basis.py.
// The stamp functions (_naht_passt, _scatter_details, _stamp_volcanoes) live
// elsewhere.
#include "mapgen.hpp"

#include <cmath>

namespace om::gen {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Python round(): half to even (the default floating-point rounding mode).
int roundHalfEven(double v) { return static_cast<int>(std::nearbyint(v)); }

}  // namespace

// Python: MapGenParams.from_template / TEMPLATES
void applyTemplate(Params& p, const std::string& name) {
    p.landscape = false;
    p.volcanoes = 0;
    if (name == "canyon") {
        p.mix = {{"roc", 0.55}, {"mud", 0.25}, {"san", 0.10}, {"veg", 0.10}};
        p.roughness = 0.75;
    } else if (name == "desert") {
        p.mix = {{"san", 0.60}, {"mud", 0.30}, {"roc", 0.08}, {"veg", 0.02}};
        p.roughness = 0.35;
    } else if (name == "islands") {
        p.mix = {{"mud", 0.35}, {"roc", 0.35}, {"san", 0.20}, {"veg", 0.10}};
        p.roughness = 0.85;
    } else if (name == "ice") {
        p.mix = {{"sno", 0.45}, {"roc", 0.35}, {"mud", 0.20}};
        p.roughness = 0.45;
    } else if (name == "newterra") {
        // Landscape mode: relief instead of rolled patches; small sand share on purpose.
        p.mix = {{"mud", 0.68}, {"san", 0.20}, {"roc", 0.12}};
        p.roughness = 0.0;
        p.landscape = true;
        p.volcanoes = 1;
    } else {   // plains, and unknown names
        p.mix = {{"mud", 0.45}, {"roc", 0.25}, {"san", 0.15}, {"veg", 0.15}};
        p.roughness = 0.0;
    }
}

// Python: _value_noise
Floats valueNoise(Rng& rng, int h, int w, double freq) {
    const int f = std::max(2, static_cast<int>(freq));   // int() truncates
    std::vector<double> base(size_t(f + 1) * (f + 1));
    for (double& b : base) b = rng.uniform();
    auto lattice = [&](int y, int x) { return base[size_t(y) * (f + 1) + x]; };
    auto smoothstep = [](double t) { return t * t * (3 - 2 * t); };

    // linspace(0, freq, n, endpoint=False): sample i sits at i * freq / n.
    std::vector<int> x0(size_t(std::max(w, 0)));
    std::vector<double> fx(x0.size());
    for (int x = 0; x < w; ++x) {
        const double s = double(x) * f / w;
        x0[x] = static_cast<int>(s);
        fx[x] = smoothstep(s - x0[x]);
    }
    Floats out(w, h);
    for (int y = 0; y < h; ++y) {
        const double s = double(y) * f / h;
        const int y0 = static_cast<int>(s);
        const double fy = smoothstep(s - y0);
        for (int x = 0; x < w; ++x) {
            const double top = lattice(y0, x0[x]) * (1 - fx[x]) + lattice(y0, x0[x] + 1) * fx[x];
            const double bottom = lattice(y0 + 1, x0[x]) * (1 - fx[x]) + lattice(y0 + 1, x0[x] + 1) * fx[x];
            out.at(x, y) = top * (1 - fy) + bottom * fy;
        }
    }
    return out;
}

// Python: _family_field
Field familyField(const Params& p, Rng& rng, const std::vector<std::string>& groundIds) {
    const int h = p.height, w = p.width;
    // Roughness controls the frequency: few large areas vs. many small ones.
    const double baseFreq = 3 + p.roughness * 14;
    double total = 0;
    for (const auto& m : p.mix)
        if (m.second > 0) total += m.second;
    if (total == 0) total = 1.0;

    std::vector<Floats> scores;
    std::vector<int16_t> order;
    std::vector<double> targets;
    for (const auto& m : p.mix) {
        if (m.second <= 0) continue;
        const auto it = std::find(groundIds.begin(), groundIds.end(), m.first);
        if (it == groundIds.end()) continue;
        Floats a = valueNoise(rng, h, w, baseFreq);
        const Floats b = valueNoise(rng, h, w, baseFreq * 2);
        for (size_t i = 0; i < a.v.size(); ++i) a.v[i] = (a.v[i] + 0.5 * b.v[i]) / 1.5;
        scores.push_back(std::move(a));
        order.push_back(static_cast<int16_t>(it - groundIds.begin()));
        targets.push_back(m.second / total);
    }
    if (scores.empty()) return Field(w, h, 0);

    // Tune a per-family offset until the area shares match the request; a fixed
    // bonus per weight misses the target because the noise fields overlap differently.
    const size_t n = scores.size(), cells = size_t(w) * h;
    std::vector<double> offsets(n, 0.0), shares(n);
    std::vector<int> picked(cells, 0);
    for (int iter = 0; iter < 40; ++iter) {
        std::vector<size_t> counts(n, 0);
        for (size_t c = 0; c < cells; ++c) {
            int best = 0;   // argmax: first maximum wins
            double bestV = scores[0].v[c] + offsets[0];
            for (size_t k = 1; k < n; ++k) {
                const double v = scores[k].v[c] + offsets[k];
                if (v > bestV) { bestV = v; best = int(k); }
            }
            picked[c] = best;
            ++counts[size_t(best)];
        }
        double maxErr = 0;
        std::vector<double> error(n);
        for (size_t k = 0; k < n; ++k) {
            shares[k] = cells ? double(counts[k]) / double(cells) : 0.0;
            error[k] = targets[k] - shares[k];
            maxErr = std::max(maxErr, std::abs(error[k]));
        }
        if (maxErr < 0.015) break;
        for (size_t k = 0; k < n; ++k) offsets[k] += 0.4 * error[k];
    }
    Field out(w, h, 0);
    for (size_t c = 0; c < cells; ++c) out.v[c] = order[size_t(picked[c])];
    return out;
}

// Python: _smooth_families (np.roll, so the neighbourhood wraps around the map edges)
Field smoothFamilies(const Field& f, int passes, int radius) {
    Field out = f;
    const int w = f.w, h = f.h;
    if (w <= 0 || h <= 0) return out;
    int kinds = 0;
    for (int16_t v : f.v) kinds = std::max(kinds, int(v) + 1);
    if (kinds <= 0) return out;
    std::vector<int> counts(static_cast<size_t>(kinds), 0);
    for (int pass = 0; pass < std::max(0, passes); ++pass) {
        Field next(w, h, 0);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                std::fill(counts.begin(), counts.end(), 0);
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int yy = ((y - dy) % h + h) % h;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int xx = ((x - dx) % w + w) % w;
                        const int v = out.at(xx, yy);
                        if (v >= 0 && v < kinds) ++counts[size_t(v)];
                    }
                }
                int best = 0;   // argmax: lowest family wins ties
                for (int k = 1; k < kinds; ++k)
                    if (counts[size_t(k)] > counts[size_t(best)]) best = k;
                next.at(x, y) = static_cast<int16_t>(best);
            }
        }
        out = std::move(next);
    }
    return out;
}

// Python: _components
std::vector<std::vector<int>> components(const Mask& m) {
    std::vector<std::vector<int>> result;
    const int w = m.w, h = m.h;
    std::vector<uint8_t> seen(m.v.size(), 0);
    std::vector<int> stack;
    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            const int s = sy * w + sx;
            if (!m.v[size_t(s)] || seen[size_t(s)]) continue;
            std::vector<int> part;
            stack.assign(1, s);
            seen[size_t(s)] = 1;
            while (!stack.empty()) {
                const int c = stack.back();
                stack.pop_back();
                part.push_back(c);
                const int x = c % w, y = c / w;
                static const int kDir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (const auto& d : kDir) {
                    const int nx = x + d[0], ny = y + d[1];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const int nc = ny * w + nx;
                    if (m.v[size_t(nc)] && !seen[size_t(nc)]) {
                        seen[size_t(nc)] = 1;
                        stack.push_back(nc);
                    }
                }
            }
            result.push_back(std::move(part));
        }
    }
    return result;
}

// Python: _dissolve_small_biomes (patches are found on the input field, replaced in a copy)
void dissolveSmallBiomes(Field& f, int minCells) {
    if (minCells <= 1) return;
    const Field src = f;
    const int w = src.w, h = src.h;
    std::vector<uint8_t> seen(src.v.size(), 0);
    std::vector<int> stack, cells, rim;
    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            if (seen[size_t(sy) * w + sx]) continue;
            const int16_t fam = src.at(sx, sy);
            stack.assign(1, sy * w + sx);
            seen[size_t(sy) * w + sx] = 1;
            cells.clear();
            rim.clear();   // foreign neighbour families, one entry per touching edge
            while (!stack.empty()) {
                const int c = stack.back();
                stack.pop_back();
                cells.push_back(c);
                const int x = c % w, y = c / w;
                static const int kDir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (const auto& d : kDir) {
                    const int nx = x + d[0], ny = y + d[1];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const int nc = ny * w + nx;
                    if (src.v[size_t(nc)] == fam) {
                        if (!seen[size_t(nc)]) {
                            seen[size_t(nc)] = 1;
                            stack.push_back(nc);
                        }
                    } else {
                        rim.push_back(src.v[size_t(nc)]);
                    }
                }
            }
            if (int(cells.size()) >= minCells || rim.empty()) continue;
            // Most common neighbour; ties go to the smallest family (np.unique + argmax).
            std::sort(rim.begin(), rim.end());
            int best = rim[0], bestN = 0;
            for (size_t i = 0; i < rim.size();) {
                size_t j = i;
                while (j < rim.size() && rim[j] == rim[i]) ++j;
                if (int(j - i) > bestN) { bestN = int(j - i); best = rim[i]; }
                i = j;
            }
            for (int c : cells) f.v[size_t(c)] = static_cast<int16_t>(best);
        }
    }
}

// Python: _apply_symmetry - the layout is mirrored, not the tile graphics.
// For even sizes this is numpy's slicing exactly; for odd sizes (where the
// Python slices don't match and raise) the middle row/column stays as it is.
template <class T>
void applySymmetry(Grid<T>& g, Sym s) {
    const int w = g.w, h = g.h;
    auto mirrorX = [&] {
        for (int y = 0; y < h; ++y)
            for (int x = w - w / 2; x < w; ++x) g.at(x, y) = g.at(w - 1 - x, y);
    };
    auto mirrorY = [&] {
        for (int y = h - h / 2; y < h; ++y)
            for (int x = 0; x < w; ++x) g.at(x, y) = g.at(x, h - 1 - y);
    };
    switch (s) {
    case Sym::MirrorX: mirrorX(); break;
    case Sym::MirrorY: mirrorY(); break;
    case Sym::Rot180:
        for (int y = h - h / 2; y < h; ++y)
            for (int x = 0; x < w; ++x) g.at(x, y) = g.at(w - 1 - x, h - 1 - y);
        break;
    case Sym::Quad: mirrorX(); mirrorY(); break;
    case Sym::None: break;
    }
}

template void applySymmetry<uint8_t>(Grid<uint8_t>&, Sym);
template void applySymmetry<int16_t>(Grid<int16_t>&, Sym);
template void applySymmetry<int32_t>(Grid<int32_t>&, Sym);
template void applySymmetry<double>(Grid<double>&, Sym);

// Python: _mirror_region - placed formations follow the mirrored regions.
void mirrorRegion(Grid<int32_t>& tiles, Sym s) { applySymmetry(tiles, s); }

// Python: _bewuchs_abstand (mapgen_uebergaenge.py). Vegetation is mossy mud in
// OP2 with no transitions to rock or sand, so mud is pushed in as a buffer.
int vegetationSpacing(Field& f, int veg, int mud, int spacing) {
    if (veg < 0 || mud < 0 || spacing <= 0) return 0;
    const int w = f.w, h = f.h;
    bool anyVeg = false;
    for (int16_t v : f.v) anyVeg = anyVeg || v == veg;
    if (!anyVeg) return 0;
    // Foreign is everything except vegetation and mud.
    Mask near(w, h, 0);
    for (size_t i = 0; i < f.v.size(); ++i) near.v[i] = (f.v[i] != veg && f.v[i] != mud) ? 1 : 0;
    for (int step = 0; step < spacing; ++step) {
        Mask grown = near;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if (near.get(x - 1, y, 0) || near.get(x + 1, y, 0) || near.get(x, y - 1, 0) || near.get(x, y + 1, 0))
                    grown.at(x, y) = 1;
        near = std::move(grown);
    }
    int hits = 0;
    for (size_t i = 0; i < f.v.size(); ++i) {
        if (f.v[i] == veg && near.v[i]) {
            f.v[i] = static_cast<int16_t>(mud);
            ++hits;
        }
    }
    return hits;
}

// Python: _volcano_sites. Volcanoes sit in rock: the surroundings get the rock
// family with an irregular rim. The lava comes from the eruption in-game.
std::vector<Point> volcanoSites(Field& f, const Params& p, Rng& rng, int rockFamily) {
    std::vector<Point> centres;
    if (p.volcanoes <= 0) return centres;
    const int w = f.w, h = f.h;
    const int radius = std::max(6, roundHalfEven(std::sqrt(p.lavaZoneTiles / kPi)));
    const int margin = radius + 4;
    if (w <= 2 * margin || h <= 2 * margin) return centres;

    auto mirrored = [&](int cx, int cy) -> std::vector<Point> {
        switch (p.symmetry) {
        case Sym::MirrorX: return {{w - 1 - cx, cy}};
        case Sym::MirrorY: return {{cx, h - 1 - cy}};
        case Sym::Rot180: return {{w - 1 - cx, h - 1 - cy}};
        case Sym::Quad: return {{w - 1 - cx, cy}, {cx, h - 1 - cy}, {w - 1 - cx, h - 1 - cy}};
        case Sym::None: break;
        }
        return {};
    };

    const int wanted = p.volcanoes;
    for (int guard = 0; int(centres.size()) < wanted && guard < 40; ++guard) {
        const int cx = int(rng.integers(margin, w - margin));
        const int cy = int(rng.integers(margin, h - margin));
        bool tooClose = false;
        for (const Point& o : centres) {
            const long long dx = cx - o.first, dy = cy - o.second;
            if (dx * dx + dy * dy < 9LL * radius * radius) tooClose = true;
        }
        if (tooClose) continue;
        std::vector<Point> group{{cx, cy}};
        for (const Point& m : mirrored(cx, cy)) group.push_back(m);
        for (const Point& g : group) {
            if (int(centres.size()) >= wanted) break;
            const Floats noise = valueNoise(rng, h, w, 8);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const double wobble = 0.75 + 0.5 * noise.at(x, y);
                    const double dist = std::hypot(double(x - g.first), double(y - g.second));
                    if (dist < radius * wobble) f.at(x, y) = static_cast<int16_t>(rockFamily);
                }
            }
            centres.push_back(g);
        }
    }
    return centres;
}

// Python: _lava_zone. Not terrain but a mission fact (SetLavaPossible); row-major order.
std::vector<Point> lavaZone(Point centre, int w, int h, int tiles, Rng& rng) {
    const int radius = std::max(4, roundHalfEven(std::sqrt(tiles / kPi)));
    const Floats noise = valueNoise(rng, h, w, 10);
    std::vector<Point> out;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double wobble = 0.7 + 0.6 * noise.at(x, y);
            const double dist = std::hypot(double(x - centre.first), double(y - centre.second));
            if (dist < radius * wobble) out.emplace_back(x, y);
        }
    }
    return out;
}

}  // namespace om::gen
