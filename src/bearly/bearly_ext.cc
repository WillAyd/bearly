#include <nanoarrow/nanoarrow.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <map>

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
  ArrowError error;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayStreamGetSchema(stream.get(), schema.get(), &error));

  // identify which columns are even summable
  using nchildren_t = decltype(schema->n_children);
  std::map<nchildren_t, int64_t> sum_results;
  for (nchildren_t i = 0; i < schema->n_children; ++i) {
    ArrowSchemaView schema_view;
    NANOARROW_THROW_NOT_OK(
        ArrowSchemaViewInit(&schema_view, schema->children[i], &error));
    switch (schema_view.type) {
    case NANOARROW_TYPE_INT64:
      sum_results.emplace(i, 0);
      break;
    default:
      break;
    }
  }

  // form our result schema
  nanoarrow::UniqueSchema result_schema;
  ArrowSchemaInit(result_schema.get());
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetTypeStruct(result_schema.get(), sum_results.size()));
  for (nchildren_t result_i = 0; const auto &[key, val] : sum_results) {
    ArrowSchemaView schema_view;
    NANOARROW_THROW_NOT_OK(
        ArrowSchemaViewInit(&schema_view, schema->children[key], &error));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(result_schema->children[result_i],
                                              schema_view.type));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(result_schema->children[result_i],
                                              schema->children[key]->name));
    ++result_i;
  }

  ArrowErrorCode code;
  nanoarrow::ViewArrayStream array_stream{stream.get(), &code, &error};

  for (const auto &chunk : array_stream) {
    for (const auto &[key, val] : sum_results) {
      nanoarrow::UniqueArrayView array_view;
      ArrowArrayViewInitFromSchema(array_view.get(), schema->children[key],
                                   &error);
      NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(
          array_view.get(), chunk.children[key], &error));

      int64_t array_view_sum = 0;
      for (const auto &value :
           nanoarrow::ViewArrayAs<int64_t>(array_view.get())) {
        array_view_sum += value.value_or(0);
      }
      sum_results[key] += array_view_sum;
    }
  }

  // form output array
  nanoarrow::UniqueArray result_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(result_array.get(),
                                                  result_schema.get(), &error));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(result_array.get()));
  for (nchildren_t i = 0; const auto &[key, val] : sum_results) {
    ArrowArrayAppendInt(result_array->children[i], val);
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

auto ProduceArray() {
  // Create schema
  nanoarrow::UniqueSchema schema;
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema.get(), "awesome_numbers"));

  // Create the array
  nanoarrow::UniqueArray array;
  ArrowError error;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(array.get(), schema.get(), &error));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(array.get()));

  NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array.get(), 42));
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array.get(), 555));
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(array.get(), 1));

  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(array.get(), &error));

  // Expose those as capsules
  auto c_schema = static_cast<ArrowSchema *>(malloc(sizeof(ArrowSchema)));
  schema.move(c_schema);
  constexpr auto schemaCleanup = [](void *ptr) noexcept {
    auto schema = static_cast<ArrowSchema *>(ptr);
    if (schema->release != nullptr) {
      schema->release(schema);
    }
    free(schema);
  };
  nb::capsule schema_capsule{c_schema, "arrow_schema", schemaCleanup};

  auto c_array = static_cast<ArrowArray *>(malloc(sizeof(ArrowArray)));
  array.move(c_array);
  constexpr auto arrayCleanup = [](void *ptr) noexcept {
    auto array = static_cast<ArrowArray *>(ptr);
    if (array->release != nullptr) {
      array->release(array);
    }
    free(array);
  };
  nb::capsule array_capsule{c_array, "arrow_array", arrayCleanup};

  return nb::make_tuple(schema_capsule, array_capsule);
}

auto ProduceStream() {
  // Create our schema
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 2));
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "column0"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "column1"));

  // Create our array
  nanoarrow::UniqueArray array;
  ArrowError error;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(array.get(), schema.get(), &error));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(array.get()));

  // row 0
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array->children[0], 42));
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array->children[1], 555));
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(array.get()));

  // row 1
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array->children[0], 84));
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array->children[1], 1110));
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(array.get()));

  // row 2
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(array->children[0], 1));
  NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(array->children[1], 1));
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(array.get()));

  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(array.get(), &error));

  // Exchange as capsules
  auto c_stream =
      static_cast<ArrowArrayStream *>(malloc(sizeof(ArrowArrayStream)));
  if (ArrowBasicArrayStreamInit(c_stream, schema.get(), 1)) {
    free(c_stream);
    throw "ArrowBasicArrayStreamInit call failed!";
  }

  ArrowBasicArrayStreamSetArray(c_stream, 0, array.get());
  constexpr auto streamRelease = [](void *ptr) noexcept {
    auto stream = static_cast<ArrowArrayStream *>(ptr);
    if (stream->release != nullptr) {
      stream->release(stream);
    }

    free(stream);
  };

  return nb::capsule{c_stream, "arrow_array_stream", streamRelease};
}

NB_MODULE(bearly, m) {
  m.def("sum", &SumChunks);
  m.def("produce_array", &ProduceArray);
  m.def("produce_stream", &ProduceStream);
}
