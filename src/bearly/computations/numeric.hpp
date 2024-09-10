#pragma once

#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <nanoarrow/nanoarrow.hpp>

#include "../types.hpp"

template <typename T>
concept Sumable = std::integral<typename T::c_type> ||
                  std::floating_point<typename T::c_type>;

template <typename T>
concept FixedSizePrimitive =
std::is_same_v<T, bearly::BooleanType> || std::is_same_v<T, bearly::Int8Type> ||
    std::is_same_v<T, bearly::UInt8Type> || std::is_same_v<T, bearly::Int16Type> ||
    std::is_same_v<T, bearly::UInt16Type> || std::is_same_v<T, bearly::Int32Type> ||
    std::is_same_v<T, bearly::UInt32Type> || std::is_same_v<T, bearly::Int64Type> ||
    std::is_same_v<T, bearly::UInt64Type> || std::is_same_v<T, bearly::FloatType> ||
    std::is_same_v<T, bearly::DoubleType>;

template <typename T>
concept VariableSizeBinary =
    std::is_same_v<T, bearly::StringType> || std::is_same_v<T, bearly::BinaryType>;

struct SumVisitor {
  explicit SumVisitor(const ArrowArrayView *array_view) : array_view_(array_view) {}

  template <typename T>
  auto operator()(const T)
    requires FixedSizePrimitive<T>
  {
    using ResultT = std::conditional_t<std::is_unsigned_v<typename T::c_type>, uint64_t, int64_t>;
    const auto vw_rng = nanoarrow::ViewArrayAs<typename T::c_type>(array_view_);
    constexpr auto zero_val = ResultT{};
    constexpr auto func = [](auto &&a, const auto &&b) {
      return a += b.value_or(zero_val);
    };

    const auto val =
        std::accumulate(vw_rng.begin(), vw_rng.end(), zero_val, func);

    return bearly::Scalar{val};
  }

  template <typename T> auto operator()(const T) {
    return bearly::Scalar{std::monostate{}};
  }
  
private:
  const ArrowArrayView *array_view_;
};


/*
  struct MinVisitor {
  explicit MinVisitor(const ArrowArrayView *array_view) :
array_view_(array_view) {}

  template <typename T>
  auto operator()(const T)
    requires FixedSizePrimitive<T>
  {
    const auto vw_rng = nanoarrow::ViewArrayAs<typename T::c_type>(array_view_);
    constexpr auto func = [](const auto &&a, const auto &&b) {
      return std::min(a, b.value_or(typename T::c_type{}));
    };

    const auto val = std::accumulate(vw_rng.begin(), vw_rng.end(), typename T::c_type{}, func);
    return bearly::Scalar(T{}, val);
  }

  template <typename T>
  auto operator()(const T)
    requires VariableSizeBinary<T>
  {
    const auto vw_rng = nanoarrow::ViewArrayAsBytes<T::OffsetSize>(array_view_);
    constexpr auto func = [](const auto &&a, const auto &&b) {
      // TODO: probably want a real unicode library, but this works for now
      return ArrowStringView{"foo", 3};
    };

    const auto val = std::accumulate(vw_rng.begin(), vw_rng.end(), ArrowStringView{"foo", 3}, func);
    return bearly::Scalar(T{}, val);
  }

  template <typename T> auto operator()(const T) {
    return bearly::Scalar{std::monostate{}, std::monostate{}};
  }

private:
  const ArrowArrayView *array_view_;
};
*/
