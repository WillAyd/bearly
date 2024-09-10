#pragma once

#include <nanoarrow/nanoarrow.hpp>
#include <variant>

namespace bearly {
struct UninitializedType {};
struct NAType {};
struct BooleanType {
  static constexpr ArrowType arrow_type_ = NANOARROW_TYPE_NA;
  using c_type = bool;
};
struct UInt8Type {
  using c_type = uint8_t;
};
struct Int8Type {
  using c_type = int8_t;  
};
struct UInt16Type {
  using c_type = uint16_t;
};
struct Int16Type {
  using c_type = int16_t;  
};
struct UInt32Type {
  using c_type = uint32_t;  
};
struct Int32Type {
  using c_type = int32_t;  
};
struct UInt64Type {
  using c_type = uint64_t;  
};
struct Int64Type {
  using c_type = int64_t;
};
struct HalfFloatType {
  //using c_type = float16_t;
};
struct FloatType {
  using c_type = float;
};
struct DoubleType {
  using c_type = double;
};
struct StringType {
  static constexpr int OffsetSize = 32;
};
struct BinaryType {
  static constexpr int OffsetSize = 32;
};

using Type =
  std::variant<UninitializedType, NAType, BooleanType, UInt8Type, Int8Type, UInt16Type,
                 Int16Type, UInt32Type, Int32Type, UInt64Type, Int64Type, HalfFloatType,
                 FloatType, DoubleType, StringType, BinaryType, std::monostate>;

using Scalar = std::variant <bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                             uint64_t, int64_t,
                             float, double, std::monostate>;

// https://stackoverflow.com/a/60567091/621736
template <size_t I = 0>
constexpr auto TypeVariantFromArrowType(size_t index) -> Type {
  if constexpr (I >= std::variant_size_v<Type>) {
    return std::monostate{};
  } else {
    return index == 0 ? Type{std::in_place_index<I>} :
      TypeVariantFromArrowType<I+1>(--index);
  }
};

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

auto ArrowTypeVisitor = overloaded {
  [](auto) { return NANOARROW_TYPE_UNINITIALIZED; },  // TOOD: should we?
  [](bool) { return NANOARROW_TYPE_BOOL; },
  [](uint8_t) { return NANOARROW_TYPE_UINT8; },
  [](int8_t) { return NANOARROW_TYPE_INT8; },
  [](uint16_t) { return NANOARROW_TYPE_UINT16; },
  [](int16_t) { return NANOARROW_TYPE_INT16; },
  [](uint32_t) { return NANOARROW_TYPE_UINT32; },
  [](int32_t) { return NANOARROW_TYPE_INT32; },
  [](uint64_t) { return NANOARROW_TYPE_UINT64; },
  [](int64_t) { return NANOARROW_TYPE_INT64; },
  [](float) { return NANOARROW_TYPE_FLOAT; },
  [](double) { return NANOARROW_TYPE_DOUBLE; },
};

auto ArrowArrayAppendVisitor = overloaded{
    [](auto, ArrowArray *) { return EINVAL; },
    [](bool val, ArrowArray *array) { return ArrowArrayAppendInt(array, val); },
    [](uint8_t val, ArrowArray *array) {
      return ArrowArrayAppendUInt(array, val);
    },
    [](int8_t val, ArrowArray *array) {
      return ArrowArrayAppendInt(array, val);
    },
    [](uint16_t val, ArrowArray *array) {
      return ArrowArrayAppendUInt(array, val);
    },
    [](int16_t val, ArrowArray *array) {
      return ArrowArrayAppendInt(array, val);
    },
    [](uint32_t val, ArrowArray *array) {
      return ArrowArrayAppendUInt(array, val);
    },
    [](int32_t val, ArrowArray *array) {
      return ArrowArrayAppendInt(array, val);
    },
    [](uint64_t val, ArrowArray *array) {
      return ArrowArrayAppendUInt(array, val);
    },
    [](int64_t val, ArrowArray *array) {
      return ArrowArrayAppendInt(array, val);
    },
    [](float val, ArrowArray *array) {
      return ArrowArrayAppendDouble(array, val);
    },
    [](double val, ArrowArray *array) {
      return ArrowArrayAppendDouble(array, val);
    },
};

// Higher order function to allow std::visit with arguments courtesty of
// ChatGPT. Another option would be to create a dummy variant for ArrowArray
// but I felt this was more generic
template <typename Visitor, typename Variant, typename... Args>
void visit_with_args(Visitor &&visitor, Variant &&variant, Args &&...args) {
  std::visit([&](auto &&v) {
    visitor(std::forward<decltype(v)>(v), std::forward<Args>(args)...);
  }, std::forward<Variant>(variant));
}

} // namespace bearly


