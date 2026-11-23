// Profile loading - turns the JSON profile (docs/profile-format.md) into lookup tables.
//
// Profile::fromJson validates the document and builds the tables BlendMap
// needs: tile lookup by (tileset, image), the fill tiles of every ground and
// the transition tiles of every ground pair by corner key. The optional
// "cliffs" section is handed to CliffData::parse (cliff.cpp).
#include "blend.hpp"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace om {

namespace {

// Lower-case ASCII copy, for case-insensitive tileset names.
std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Packs (tileset, image) into one key for tileIndex_.
uint64_t tileKey(int tileset, int image) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(tileset)) << 32) | static_cast<uint32_t>(image);
}

/// "abba" (NW NE SE SW) -> CornerBit mask.
uint8_t parseCorners(const std::string& s) {
    if (s.size() != 4) throw std::runtime_error("\"corners\" must have 4 letters (NW NE SE SW)");
    static const uint8_t bits[4] = {kNW, kNE, kSE, kSW};
    uint8_t mask = 0;
    for (int i = 0; i < 4; ++i) {
        if (s[i] == 'b') mask |= bits[i];
        else if (s[i] != 'a') throw std::runtime_error("\"corners\" letters must be 'a' or 'b'");
    }
    return mask;
}

}  // namespace

// Every problem (bad JSON, missing or mistyped field, unknown ground or
// tileset, bad corner key) throws std::runtime_error with a readable message;
// nothing is returned half-built.
Profile Profile::fromJson(const char* text, size_t len) {
    const json::Value doc = json::Value::parse(text, len);

    if (doc.at("format").string() != "op2maplib.blend-profile")
        throw std::runtime_error("not an op2maplib blend profile (\"format\" field)");
    if (doc.at("version").integer() != 1)
        throw std::runtime_error("unsupported profile version (this library reads version 1)");

    Profile p;
    if (const json::Value* n = doc.find("name")) p.name = n->string();

    for (const json::Value& t : doc.at("tilesets").array()) p.tilesets.push_back(lower(t.string()));

    for (const json::Value& g : doc.at("grounds").array()) {
        Ground ground;
        ground.id = g.at("id").string();
        ground.name = g.find("name") ? g.at("name").string() : ground.id;
        if (p.findGround(ground.id) >= 0) throw std::runtime_error("duplicate ground \"" + ground.id + "\"");
        p.grounds.push_back(std::move(ground));
    }
    const int groundCount = static_cast<int>(p.grounds.size());
    // One slot of 16 corner-key lists per ordered pair (a, b).
    p.byPair_.assign(static_cast<size_t>(groundCount) * groundCount, {});

    // Ground id in a tile entry -> ground index; must refer to a declared ground.
    auto groundRef = [&](const json::Value& v) {
        int g = p.findGround(v.string());
        if (g < 0) throw std::runtime_error("unknown ground \"" + v.string() + "\"");
        return g;
    };

    for (const json::Value& t : doc.at("tiles").array()) {
        ProfileTile tile;
        tile.tileset = static_cast<int>(t.at("ts").integer());
        tile.image = static_cast<int>(t.at("img").integer());
        if (tile.tileset < 0 || tile.tileset >= static_cast<int>(p.tilesets.size()))
            throw std::runtime_error("tile refers to an unknown tileset index");
        // Optional fields: weight (at least 1) and suggested cell type.
        if (const json::Value* w = t.find("w")) tile.weight = static_cast<uint32_t>(std::max<int64_t>(1, w->integer()));
        if (const json::Value* c = t.find("cell"); c && c->isNumber()) tile.cellType = static_cast<int>(c->integer());

        const std::string& role = t.at("role").string();
        if (role == "plain" || role == "ground") {
            tile.role = role == "plain" ? TileRole::Fill : TileRole::Ground;
            tile.ground = groundRef(t.at("ground"));
        } else if (role == "transition") {
            tile.role = TileRole::Transition;
            tile.ground = groundRef(t.at("a"));
            tile.groundB = groundRef(t.at("b"));
            tile.corners = parseCorners(t.at("corners").string());
        } else {
            throw std::runtime_error("unknown tile role \"" + role + "\"");
        }

        const uint64_t key = tileKey(tile.tileset, tile.image);
        if (p.tileIndex_.count(key)) continue;   // first entry wins; duplicates are ignored
        const int index = static_cast<int>(p.tiles.size());
        p.tileIndex_[key] = index;
        // Index the tile: only fill tiles are placed on plain ground, and only
        // transition tiles are looked up by pair and corner key.
        if (tile.role == TileRole::Fill) p.grounds[tile.ground].fill.push_back(index);
        if (tile.role == TileRole::Transition)
            p.byPair_[static_cast<size_t>(tile.ground) * groundCount + tile.groundB][tile.corners].push_back(index);
        p.tiles.push_back(tile);
    }

    // Optional standard mapping table: [tileset index, image] per mapping index.
    if (const json::Value* sm = doc.find("standardMapping")) {
        for (const json::Value& e : sm->array()) {
            const json::Array& pair = e.array();
            if (pair.size() != 2) throw std::runtime_error("\"standardMapping\" entries must be [tileset, image]");
            const int ts = static_cast<int>(pair[0].integer());
            if (ts < 0 || ts >= static_cast<int>(p.tilesets.size()))
                throw std::runtime_error("\"standardMapping\" refers to an unknown tileset index");
            p.standardMapping.emplace_back(ts, static_cast<int>(pair[1].integer()));
        }
    }

    if (const json::Value* c = doc.find("cliffs")) p.cliffs.parse(*c, static_cast<int>(p.tilesets.size()));
    if (const json::Value* g = doc.find("generator")) parseGenerator(p, *g);
    return p;
}

// The optional "generator" section: detail and volcano stamps, and the packed
// tile pairs seen side by side (flat arrays a0, b0, a1, b1, ...).
void Profile::parseGenerator(Profile& p, const json::Value& g) {
    const int tilesetCount = static_cast<int>(p.tilesets.size());
    auto stamps = [&](const char* key, std::vector<Stamp>& out) {
        const json::Value* list = g.find(key);
        if (!list) return;
        for (const json::Value& s : list->array()) {
            Stamp st;
            if (const json::Value* n = s.find("name")) st.name = n->string();
            if (const json::Value* gr = s.find("ground"); gr && !gr->string().empty()) {
                st.ground = p.findGround(gr->string());
                if (st.ground < 0) throw std::runtime_error("generator stamp: unknown ground \"" + gr->string() + "\"");
            }
            st.w = static_cast<int>(s.at("w").integer());
            st.h = static_cast<int>(s.at("h").integer());
            for (const json::Value& t : s.at("tiles").array()) {
                const json::Array& a = t.array();
                if (a.size() != 3) throw std::runtime_error("generator stamp tiles must be [tileset, image, cell]");
                StampTile tile{static_cast<int>(a[0].integer()), static_cast<int>(a[1].integer()),
                               static_cast<int>(a[2].integer())};
                if (tile.tileset < 0 || tile.tileset >= tilesetCount)
                    throw std::runtime_error("generator stamp refers to an unknown tileset index");
                st.tiles.push_back(tile);
            }
            if (st.w <= 0 || st.h <= 0 || static_cast<int>(st.tiles.size()) != st.w * st.h)
                throw std::runtime_error("generator stamp: \"tiles\" must hold w * h entries");
            out.push_back(std::move(st));
        }
    };
    stamps("details", p.generator.details);
    stamps("volcanoes", p.generator.volcanoes);

    auto pairs = [&](const char* key, std::unordered_map<uint64_t, uint8_t>& seen,
                     std::unordered_map<uint32_t, uint8_t>& known) {
        const json::Value* list = g.find(key);
        if (!list) return;
        const json::Array& a = list->array();
        if (a.size() % 2) throw std::runtime_error(std::string("generator \"") + key + "\" must hold pairs");
        seen.reserve(a.size() / 2);
        for (size_t i = 0; i + 1 < a.size(); i += 2) {
            const uint32_t x = static_cast<uint32_t>(a[i].integer());
            const uint32_t y = static_cast<uint32_t>(a[i + 1].integer());
            seen[(static_cast<uint64_t>(x) << 32) | y] = 1;
            known[x] = 1;
        }
    };
    pairs("right", p.generator.right, p.generator.rightKnown);
    pairs("down", p.generator.down, p.generator.downKnown);
}

bool GeneratorData::seen(bool horizontal, uint32_t a, uint32_t b) const {
    const auto& m = horizontal ? right : down;
    return m.count((static_cast<uint64_t>(a) << 32) | b) != 0;
}

bool GeneratorData::known(bool horizontal, uint32_t a) const {
    return (horizontal ? rightKnown : downKnown).count(a) != 0;
}

// Linear searches: profiles have only a handful of grounds and tilesets.
int Profile::findGround(const std::string& id) const {
    for (size_t i = 0; i < grounds.size(); ++i)
        if (grounds[i].id == id) return static_cast<int>(i);
    return -1;
}

int Profile::findTileset(const std::string& lowerName) const {
    for (size_t i = 0; i < tilesets.size(); ++i)
        if (tilesets[i] == lowerName) return static_cast<int>(i);
    return -1;
}

int Profile::findTile(int tileset, int image) const {
    auto it = tileIndex_.find(tileKey(tileset, image));
    return it == tileIndex_.end() ? -1 : it->second;
}

// Invalid ground indices (for example -1 for a fixed cell) give the empty set.
const std::array<std::vector<int>, 16>& Profile::transitions(int a, int b) const {
    const int n = static_cast<int>(grounds.size());
    if (a < 0 || b < 0 || a >= n || b >= n) return empty_;
    return byPair_[static_cast<size_t>(a) * n + b];
}

bool Profile::hasTransitions(int a, int b) const {
    for (const auto& list : transitions(a, b))
        if (!list.empty()) return true;
    return false;
}

}  // namespace om
