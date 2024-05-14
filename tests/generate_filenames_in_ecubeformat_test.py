import pytest
from datetime import datetime, timedelta
from neuraltoolkit import generate_filenames_in_ecubeformat

def test_generate_filenames_in_ecubeformat():
    initial_filename = 'Headstages_64_Channels_int16_2023-12-19_13-28-09.bin'
    expected_result = [
        'Headstages_64_Channels_int16_2023-12-19_13-33-09.bin',
        'Headstages_64_Channels_int16_2023-12-19_13-38-09.bin',
        'Headstages_64_Channels_int16_2023-12-19_13-43-09.bin',
        'Headstages_64_Channels_int16_2023-12-19_13-48-09.bin',
        'Headstages_64_Channels_int16_2023-12-19_13-53-09.bin'
    ]
    result = \
        generate_filenames_in_ecubeformat(initial_filename,
                                          total_minutes=25,
                                          interval_minutes=5)
    assert result == expected_result

