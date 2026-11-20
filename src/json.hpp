// Minimal JSON reader for profile files.
//
// Supports the full JSON grammar (objects, arrays, strings with escapes,
// numbers, true/false/null). Kept in-house so the library has no
// dependencies. Values are immutable after parsing.
//
// Used only by profile.cpp and the cliff section parser. Errors (bad syntax,
// wrong type, missing field) are reported as std::runtime_error, which the C
// API turns into the error text of om_profile_load.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace om::json {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

/// A parsed JSON value. Copying is cheap: arrays and objects are shared, not
/// deep-copied. Numbers are stored as double, as in JavaScript.
class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() = default;

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }
    bool isArray() const { return type_ == Type::Array; }
    bool isObject() const { return type_ == Type::Object; }

    // Typed accessors: throw std::runtime_error if the value has another type.
    double number() const { expect(Type::Number, "number"); return number_; }
    int64_t integer() const { expect(Type::Number, "number"); return static_cast<int64_t>(number_); }   // truncates
    bool boolean() const { expect(Type::Bool, "boolean"); return bool_; }
    const std::string& string() const { expect(Type::String, "string"); return string_; }
    const Array& array() const { expect(Type::Array, "array"); return *array_; }
    const Object& object() const { expect(Type::Object, "object"); return *object_; }

    /// Member of an object, or nullptr if absent (or this is not an object).
    const Value* find(const std::string& key) const {
        if (type_ != Type::Object) return nullptr;
        auto it = object_->find(key);
        return it == object_->end() ? nullptr : &it->second;
    }

    /// Member of an object; throws if absent.
    const Value& at(const std::string& key) const {
        const Value* v = find(key);
        if (!v) throw std::runtime_error("missing field \"" + key + "\"");
        return *v;
    }

    /// Parses one complete JSON document (text need not be terminated).
    /// Throws std::runtime_error ("JSON: ...") on any syntax error.
    static Value parse(const char* text, size_t len);

private:
    void expect(Type t, const char* what) const {
        if (type_ != t) throw std::runtime_error(std::string("expected a JSON ") + what);
    }

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0;
    std::string string_;
    std::shared_ptr<Array> array_;
    std::shared_ptr<Object> object_;

    friend class Parser;   // json.cpp; builds values field by field
};

}  // namespace om::json
