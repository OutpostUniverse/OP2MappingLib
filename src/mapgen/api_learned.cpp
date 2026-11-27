// C API of the learned map generator (om_mapgen_model_*, om_mapgen_generate_learned).
#include "../handles.hpp"
#include "learned.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <thread>

struct om_mapgen_settings {
    om::gen::Params params;
};

struct om_mapgen_result {
    om::gen::Result result;
};

struct om_mapgen_model {
    std::unique_ptr<om::gen::LearnedModel> model;
};

extern "C" {

OM_API om_mapgen_model* OM_CALL om_mapgen_model_load(const void* data, size_t len, char* err, size_t err_size) {
    om::setError(err, err_size, "");
    try {
        return new om_mapgen_model{om::gen::LearnedModel::load(static_cast<const uint8_t*>(data), len)};
    } catch (const std::exception& e) {
        om::setError(err, err_size, e.what());
    } catch (...) {
        om::setError(err, err_size, "unknown error");
    }
    return nullptr;
}

OM_API void OM_CALL om_mapgen_model_free(om_mapgen_model* model) { delete model; }

OM_API om_mapgen_result* OM_CALL om_mapgen_generate_learned(const om_profile* profile, const om_mapgen_model* model,
                                                            const om_mapgen_settings* settings, int quality, int threads,
                                                            om_progress_fn progress, void* user, char* err,
                                                            size_t err_size) {
    om::setError(err, err_size, "");
    if (!profile || !model || !settings) {
        om::setError(err, err_size, "no profile, model or settings given");
        return nullptr;
    }
    const auto& p = settings->params;
    if (p.width < 16 || p.height < 16 || p.width > 1024 || p.height > 1024 || p.width % 8 || p.height % 8) {
        om::setError(err, err_size, "the learned generator needs map sizes of 16..1024 cells that are multiples of 8");
        return nullptr;
    }
    if (quality < OM_LEARNED_SKETCH || quality > OM_LEARNED_GOOD) {
        om::setError(err, err_size, "quality must be 0 (sketch), 1 (fast) or 2 (good)");
        return nullptr;
    }
    try {
        if (threads <= 0) threads = std::max(1, std::min(8, static_cast<int>(std::thread::hardware_concurrency()) / 2));
        om::nn::Pool pool(threads);
        om::gen::ProgressFn fn;
        if (progress) fn = [&](double f) { return progress(f, user) == 0; };
        om::gen::Sketch sk = om::gen::drawSketch(*model->model, p, pool, fn);
        if (quality == OM_LEARNED_SKETCH) {
            auto* r = new om_mapgen_result{};
            r->result.sketchWall = std::move(sk.wall);
            r->result.sketchGround = std::move(sk.ground);
            return r;
        }
        return new om_mapgen_result{om::gen::paintMap(profile->profile, *model->model, p, sk, quality, pool, fn)};
    } catch (const om::gen::Cancelled&) {
        om::setError(err, err_size, "cancelled");
    } catch (const std::exception& e) {
        om::setError(err, err_size, e.what());
    } catch (...) {
        om::setError(err, err_size, "unknown error");
    }
    return nullptr;
}

OM_API int OM_CALL om_mapgen_sketch_size(const om_mapgen_result* result, int32_t* width, int32_t* height) {
    const int w = result ? result->result.sketchWall.w : 0, h = result ? result->result.sketchWall.h : 0;
    if (width) *width = w;
    if (height) *height = h;
    return w * h;
}

OM_API int OM_CALL om_mapgen_get_sketch(const om_mapgen_result* result, uint8_t* walls, uint8_t* grounds, int32_t max) {
    if (!result || max <= 0) return 0;
    const auto& sw = result->result.sketchWall.v;
    const auto& sg = result->result.sketchGround.v;
    const int n = static_cast<int>(std::min<size_t>(sw.size(), static_cast<size_t>(max)));
    for (int i = 0; i < n; ++i) {
        if (walls) walls[i] = static_cast<uint8_t>(sw[size_t(i)]);
        if (grounds) grounds[i] = static_cast<uint8_t>(sg[size_t(i)]);
    }
    return n;
}

}  // extern "C"
