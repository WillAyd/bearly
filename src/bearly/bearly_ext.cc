#include "nanoarrow/nanoarrow.h"
#include <nanoarrow/nanoarrow.hpp>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

auto SumChunks(nb::object obj) {
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
  ArrowSchemaView schema_view;
  ArrowError error;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayStreamGetSchema(stream.get(), schema.get(), &error));
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaViewInit(&schema_view, schema.get(), &error));

  ArrowErrorCode code;
  nanoarrow::ViewArrayStream array_stream{stream.get(), &code, &error};
  int64_t result = 0;

  for (const auto &chunk : array_stream) {
    for (decltype(schema->n_children) i = 0; i < schema->n_children; ++i) {
      nanoarrow::UniqueArrayView array_view;
      ArrowArrayViewInitFromSchema(array_view.get(), schema->children[i], &error);
      NANOARROW_THROW_NOT_OK(
          ArrowArrayViewSetArray(array_view.get(), chunk.children[i], &error));

      for (const auto value :
           nanoarrow::ViewArrayAs<int64_t>(array_view.get())) {
        result += value.value_or(0);
      }
    }
  }

  return result;
}

NB_MODULE(bearly, m) {
  m.def("sum", &SumChunks);
}
