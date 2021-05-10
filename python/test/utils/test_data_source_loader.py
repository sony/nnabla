import pytest
from six import BytesIO, StringIO

from nnabla.utils.data_source_loader import ResourceFileReader


@pytest.mark.parametrize("source_type", ['string', 'binaryFileHandler', 'BytesIO', 'StringIO', 'strFileHandler'])
def test_class_resource_file_reader(tmpdir, source_type):

    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join('tmp.txt')
    file_path = tmppath.strpath

    with open(file_path, 'w') as f:
        f.write('test ResourceFileReader')

    def check_status(loaded, ext):
        # check source extension
        assert loaded.ext == ext

        # check source reading
        with loaded.open() as f:
            assert f.read() == b'test ResourceFileReader'

    if source_type == 'string':
        check_status(ResourceFileReader(file_path), '.txt')
    elif source_type == 'binaryFileHandler':
        with open(file_path, 'rb') as f:
            check_status(ResourceFileReader(f), '.txt')
    elif source_type == 'BytesIO':
        with open(file_path, 'rb') as f:
            check_status(ResourceFileReader(BytesIO(f.read())), '')
    elif source_type == 'StringIO':
        with pytest.raises(ValueError):
            loaded_source = ResourceFileReader(StringIO(file_path))
        return True
    elif source_type == 'strFileHandler':
        with pytest.raises(ValueError):
            with open(file_path, 'r') as f:
                loaded_source = ResourceFileReader(f)
        return True
