#include <map>
#include <stdexcept>
#include <variant>

#include <nanoarrow/nanoarrow.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "types.hpp"
#include "computations/numeric.hpp"

namespace nb = nanobind;

auto Agg(nb::object obj) {
  const auto maybe_capsule = nb::getattr(obj, "__arrow_c_stream__")();
  const auto capsule = nb::cast<nb::capsule>(maybe_capsule);

  auto c_stream = static_cast<ArrowArrayStream *>(
      PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream"));
  if (c_stream == nullptr) {
    throw nb::value_error("Invalid PyCapsule provided");
  }
  nanoarrow::UniqueArrayStream stream{c_stream};

  // Get schema from stream
  nanoarrow::UniqueSchema schema;
  ArrowError error;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayStreamGetSchema(stream.get(), schema.get(), &error));

  using nchildren_t = decltype(schema->n_children);
  std::map<nchildren_t, bearly::Scalar> agg_results;

  ArrowErrorCode code;
  nanoarrow::ViewArrayStream array_stream{stream.get(), &code, &error};

  nanoarrow::UniqueArrayView array_view;  
  for (const auto &chunk : array_stream) {
    for (nchildren_t i = 0; i < schema->n_children; ++i) {
      ArrowSchemaView schema_view;
      NANOARROW_THROW_NOT_OK(
                             ArrowSchemaViewInit(&schema_view, schema->children[i], &error));
      const auto type = bearly::TypeVariantFromArrowType(schema_view.type);
      array_view.reset();
      ArrowArrayViewInitFromSchema(array_view.get(), schema->children[i],
                                   &error);
      NANOARROW_THROW_NOT_OK(
          ArrowArrayViewSetArray(array_view.get(), chunk.children[i], &error));

      // TODO: accumulate somehow across chunks
      agg_results[i] =
        std::visit(SumVisitor(array_view.get()), type);
    }
  }

  // Create result schema
  nanoarrow::UniqueSchema result_schema;
  ArrowSchemaInit(result_schema.get());
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetTypeStruct(result_schema.get(), agg_results.size()));
  for (nchildren_t result_i = 0; const auto &[key, val] : agg_results) {
    const auto arrow_type = std::visit(bearly::ArrowTypeVisitor, val);
    NANOARROW_THROW_NOT_OK(
                            ArrowSchemaSetType(result_schema->children[result_i], arrow_type));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(result_schema->children[result_i],
                                              std::to_string(key).c_str()));
    ++result_i;
  }

  // Create result array
  nanoarrow::UniqueArray result_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(result_array.get(),
                                                  result_schema.get(), &error));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(result_array.get()));
  for (nchildren_t i = 0; const auto &[key, val] : agg_results) {
    bearly::visit_with_args(bearly::ArrowArrayAppendVisitor, val, result_array->children[i]);
    ++i;
  }

  NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(result_array.get()));
  NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuildingDefault(result_array.get(), &error));
  
  // build the capsules
  auto result_stream =
      static_cast<ArrowArrayStream *>(malloc(sizeof(ArrowArrayStream)));
  if (ArrowBasicArrayStreamInit(result_stream, result_schema.get(), 1)) {
    free(c_stream);
    throw "ArrowBasicArrayStreamInit call failed!";
  }

  ArrowBasicArrayStreamSetArray(result_stream, 0, result_array.get());

  constexpr auto streamRelease = [](void *ptr) noexcept {
    auto stream = static_cast<ArrowArrayStream *>(ptr);
    if (stream->release != nullptr) {
      stream->release(stream);
    }
    free(stream);
  };

  return nb::capsule{result_stream, "arrow_array_stream", streamRelease};
}

NB_MODULE(bearly, m) {
  m.def("agg", &Agg);
}
