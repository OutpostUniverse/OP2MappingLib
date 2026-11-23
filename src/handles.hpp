// Opaque C handle types from the public header, shared by the API files
// (api.cpp, mapgen/api_mapgen.cpp). Each wraps the C++ object; the caller owns
// the handle and releases it with the matching *_free function. Strings handed
// out by the API point into these objects and stay valid until the handle is
// freed.
#pragma once

#include "blend.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

struct om_profile {
    om::Profile profile;
};

// The profile must outlive every blend map created from it (documented in the header).
struct om_blend_map {
    std::unique_ptr<om::BlendMap> map;
};

namespace om {

// Copies msg into the caller's error buffer, truncating to fit and always
// terminating it. Does nothing if the caller passed no buffer.
inline void setError(char* err, size_t size, const char* msg) {
    if (!err || size == 0) return;
    const size_t n = std::min(std::strlen(msg), size - 1);
    std::memcpy(err, msg, n);
    err[n] = '\0';
}

}  // namespace om
