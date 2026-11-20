// Minimal JSON reader - see json.hpp.
//
// A recursive-descent parser over a (pointer, end) range. It never reads past
// end, so the input does not have to be NUL-terminated.
#include "json.hpp"

#include <cstdlib>
#include <cstring>

namespace om::json {

/// One-shot parser; Value::parse creates one per document.
class Parser {
public:
    Parser(const char* text, size_t len) : p_(text), end_(text + len) {}

    /// The whole input must be exactly one value, with optional whitespace around it.
    Value parseDocument() {
        skipWs();
        Value v = parseValue(0);
        skipWs();
        if (p_ != end_) fail("unexpected text after the JSON value");
        return v;
    }

private:
    static constexpr int kMaxDepth = 64;   // nesting limit, so hostile input cannot overflow the stack

    [[noreturn]] void fail(const char* msg) const {
        throw std::runtime_error(std::string("JSON: ") + msg);
    }

    void skipWs() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }

    // Skips over word if the input continues with it.
    bool consume(const char* word) {
        size_t n = std::strlen(word);
        if (static_cast<size_t>(end_ - p_) >= n && std::memcmp(p_, word, n) == 0) {
            p_ += n;
            return true;
        }
        return false;
    }

    // Dispatches on the first character of a value; p_ must be on it.
    Value parseValue(int depth) {
        if (depth > kMaxDepth) fail("nested too deeply");
        if (p_ >= end_) fail("unexpected end of input");
        Value v;
        switch (*p_) {
        case '{': return parseObject(depth);
        case '[': return parseArray(depth);
        case '"':
            v.type_ = Value::Type::String;
            v.string_ = parseString();
            return v;
        case 't':
            if (!consume("true")) fail("invalid literal");
            v.type_ = Value::Type::Bool; v.bool_ = true; return v;
        case 'f':
            if (!consume("false")) fail("invalid literal");
            v.type_ = Value::Type::Bool; v.bool_ = false; return v;
        case 'n':
            if (!consume("null")) fail("invalid literal");
            return v;
        default:
            return parseNumber();
        }
    }

    // Duplicate member names: the last one wins.
    Value parseObject(int depth) {
        Value v;
        v.type_ = Value::Type::Object;
        v.object_ = std::make_shared<Object>();
        ++p_;  // {
        skipWs();
        if (p_ < end_ && *p_ == '}') { ++p_; return v; }
        for (;;) {
            skipWs();
            if (p_ >= end_ || *p_ != '"') fail("expected a member name");
            std::string key = parseString();
            skipWs();
            if (p_ >= end_ || *p_ != ':') fail("expected ':'");
            ++p_;
            skipWs();
            (*v.object_)[key] = parseValue(depth + 1);
            skipWs();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            if (p_ < end_ && *p_ == '}') { ++p_; return v; }
            fail("expected ',' or '}'");
        }
    }

    Value parseArray(int depth) {
        Value v;
        v.type_ = Value::Type::Array;
        v.array_ = std::make_shared<Array>();
        ++p_;  // [
        skipWs();
        if (p_ < end_ && *p_ == ']') { ++p_; return v; }
        for (;;) {
            skipWs();
            v.array_->push_back(parseValue(depth + 1));
            skipWs();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            if (p_ < end_ && *p_ == ']') { ++p_; return v; }
            fail("expected ',' or ']'");
        }
    }

    // Encodes one code point as UTF-8.
    static void appendUtf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    // Reads the four hex digits of a \u escape.
    uint32_t parseHex4() {
        if (end_ - p_ < 4) fail("bad \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }

    // p_ is on the opening quote. Returns the unescaped text as UTF-8.
    std::string parseString() {
        ++p_;  // opening quote
        std::string out;
        while (p_ < end_) {
            char c = *p_++;
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') { out += c; continue; }
            if (p_ >= end_) break;
            char e = *p_++;
            switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                uint32_t cp = parseHex4();
                // A high surrogate followed by \u and a low surrogate forms one code
                // point. Otherwise the surrogate is encoded on its own (and a
                // second \u escape that is not a low surrogate is dropped);
                // profiles are ASCII in practice, so this is not refined further.
                if (cp >= 0xD800 && cp <= 0xDBFF && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                    p_ += 2;
                    uint32_t lo = parseHex4();
                    if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                appendUtf8(out, cp);
                break;
            }
            default: fail("bad escape in string");
            }
        }
        fail("unterminated string");
    }

    // Scans the characters a number can contain, then lets strtod validate
    // them: anything strtod does not consume entirely is rejected.
    Value parseNumber() {
        const char* start = p_;
        if (p_ < end_ && *p_ == '-') ++p_;
        if (p_ >= end_ || !(*p_ >= '0' && *p_ <= '9')) fail("invalid value");
        while (p_ < end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '.' || *p_ == 'e' ||
                             *p_ == 'E' || *p_ == '+' || *p_ == '-')) ++p_;
        std::string text(start, p_);
        char* stop = nullptr;
        double d = std::strtod(text.c_str(), &stop);
        if (stop != text.c_str() + text.size()) fail("invalid number");
        Value v;
        v.type_ = Value::Type::Number;
        v.number_ = d;
        return v;
    }

    const char* p_;
    const char* end_;
};

Value Value::parse(const char* text, size_t len) {
    return Parser(text, len).parseDocument();
}

}  // namespace om::json
