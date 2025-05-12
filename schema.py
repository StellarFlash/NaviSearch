# pylind.disable = missing-module-docstring
schema = {
    "fields": [
        {
            "name": "id",
            "dtype": "INT64",
            "is_primary": True,
            "auto_id": True
        },
        {
            "name": "title",
            "dtype": "VARCHAR",
            "max_length": 256
        },
        {
            "name":"source",
            "dtype": "VARCHAR",
            "max_length": 256
        },
        {
            "name": "content",
            "dtype": "VARCHAR",
            "max_length": 65535
        },
        {
            "name": "tags",
            "dtype": "JSON"
        },
        {
            "name": "embedding",
            "dtype": "FLOAT_VECTOR",
            "dim": 1024
        }
    ],
    "description": "带有source的schema"
}

print(schema)