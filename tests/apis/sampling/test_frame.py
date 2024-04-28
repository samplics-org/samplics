from samplics.apis.sampling import Frame


def test_frame_dict_initialization():
    data = {
        "id": [1, 2, 3, 4, 5],
        "age": [23, 45, 67, 89, 12],
    }
    schema = None
    frame = Frame(data, schema)
    assert frame.data == data
    assert frame.schema == schema
