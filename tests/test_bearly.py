import pyarrow as pa

import bearly as bl


def test_produce_array():
    schema_capsule, array_capsule = bl.produce_array()
    result = pa.Array._import_from_c_capsule(schema_capsule, array_capsule)
    expected = pa.array([42, 555, None], type=pa.int64())
    assert result == expected

def test_produce_stream():
    capsule = bl.produce_stream()
    batch_reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
    result = batch_reader.read_all()
    expected = pa.Table.from_pydict({
        "column0": pa.array([42, 84, None], type=pa.int32()),
        "column1": [555, 1110, None],
    })

    assert result == expected

def test_sum():
    tbl = pa.Table.from_pydict({
        "col0": [1, 2, None],
        "col1": [3, None, 4],
        "col2": ["foo", "bar", "baz"],
    })

    capsule = bl.sum(tbl)
    batch_reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
    result = batch_reader.read_all()

    expected = pa.Table.from_pydict({
        "col0": [3],
        "col1": [7],
    })

    assert result == expected
