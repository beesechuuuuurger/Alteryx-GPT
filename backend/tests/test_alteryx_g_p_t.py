






from ayx_python_sdk.core import Anchor
from ayx_python_sdk.core.testing import BatchTuple, SdkToolTestService

from backend.ayx_plugins import AlteryxGPT

import pyarrow as pa
from pyarrow import RecordBatch

import pytest


TEST_SCHEMA = pa.schema([
    ('col1', pa.int64()),
    ('col2', pa.float64())
])


@pytest.fixture
def small_batches():
    input_data = [
        [1, 2, 3],
        [0.1, 0.2, 0.3]
    ]
    output_data = input_data
    return BatchTuple(
        input_data=RecordBatch.from_arrays(input_data, schema=TEST_SCHEMA),
        expected_output_data=RecordBatch.from_arrays(output_data, schema=TEST_SCHEMA)
    )


@pytest.fixture()
def medium_batches():
    repeat = 200
    input_data = [
        [1, 2, 3] * repeat,
        [0.1, 0.2, 0.3] * repeat
    ]
    output_data = input_data
    return BatchTuple(
        input_data=RecordBatch.from_arrays(input_data, schema=TEST_SCHEMA),
        expected_output_data=RecordBatch.from_arrays(output_data, schema=TEST_SCHEMA)
    )


@pytest.fixture()
def large_batches():
    repeat = 20000
    input_data = [
        [1, 2, 3] * repeat,
        [0.1, 0.2, 0.3] * repeat
    ]
    output_data = input_data
    return BatchTuple(
        input_data=RecordBatch.from_arrays(input_data, schema=TEST_SCHEMA),
        expected_output_data=RecordBatch.from_arrays(output_data, schema=TEST_SCHEMA)
    )


@pytest.fixture
def alteryx_g_p_t_plugin_service():
    """
    This fixture is where you instantiate and configure your plugin's testing service.
    Please edit input_ and output_anchor_config to reflect your tool's anchor configuration.

    Note: The config_mock parameter is meant to represent the output from the UI window.
    Currently, it only takes in an XML string, wrapped in <Configuration> tags.
    """
    return SdkToolTestService(
        plugin_class=AlteryxGPT,
        config_mock="<Configuration/>",
        input_anchor_config={
            "Input": pa.schema([]),
        },
        output_anchor_config={
           "Output": pa.schema([]),
        }
    )


def test_init(alteryx_g_p_t_plugin_service):
    """
    This function is where you should test your plugin's constructor (ie, AlteryxGPT.__init__())
    Use alteryx_g_p_t_plugin_service.plugin to reference the created plugin.

    You can also test the plugin's attributes by referencing them and checking them against expected values.
    """
    assert alteryx_g_p_t_plugin_service.plugin is not None


@pytest.mark.parametrize("record_batch_set", ["small_batches", "medium_batches", "large_batches"])
@pytest.mark.parametrize("anchor", [
     Anchor("Input", "1"),
])
def test_on_record_batch(alteryx_g_p_t_plugin_service, anchor, record_batch_set, request):
    """
    This function is where you should test your plugin's on_record_batch method.
    Use alteryx_g_p_t_plugin_service.run_on_record_batch to run the specified record batch
    through the specified input anchor.

    Once the method has run, you can compare the output data against expected values,
    by accessing the corresponding data from alteryx_g_p_t_plugin_service.data_streams.
    Use the output anchor name as the dictionary key.
    If no data was written, alteryx_g_p_t_plugin_service.data_streams will be an empty dictionary.

    You can also compare IO calls made to designer through alteryx_g_p_t_plugin_service.io_stream.
    The message type (INFO, WARN, ERROR) will be prepended to the message's text with a colon.
    If no provider.io methods were called, alteryx_g_p_t_plugin_service.io_stream will be an empty list.
    """
    input_record_batch, expected_output_record_batch = request.getfixturevalue(record_batch_set)
    alteryx_g_p_t_plugin_service.run_on_record_batch(input_record_batch, anchor)
    
    assert alteryx_g_p_t_plugin_service.data_streams["Output"] == [expected_output_record_batch]
    assert alteryx_g_p_t_plugin_service.io_stream == []

    

@pytest.mark.parametrize("anchor", [
     Anchor("Input", "1"),
])
def test_on_incoming_connection_complete(alteryx_g_p_t_plugin_service, anchor):
    """
    This function is where you should test your plugin's on_incoming_connection_complete method.
    Use alteryx_g_p_t_plugin_service.run_on_incoming_connection_complete against the specified input anchors.

    Once the method has run, you can compare the output data against expected values,
    by accessing the corresponding data from alteryx_g_p_t_plugin_service.data_streams.
    Use the output anchor name as the dictionary key.
    If no data was written, alteryx_g_p_t_plugin_service.data_streams will be an empty dictionary.

    You can also compare IO calls made to designer through alteryx_g_p_t_plugin_service.io_stream.
    The message type (INFO, WARN, ERROR) will be prepended to the message's text with a colon.
    If no provider.io methods were called, alteryx_g_p_t_plugin_service.io_stream will be an empty list.
    """
    alteryx_g_p_t_plugin_service.run_on_incoming_connection_complete(anchor)
    
    assert alteryx_g_p_t_plugin_service.data_streams == {}
    assert alteryx_g_p_t_plugin_service.io_stream == [
        f"INFO:Received complete update from {anchor.name}:{anchor.connection}."
    ]

    

def test_on_complete(alteryx_g_p_t_plugin_service):
    """
    This function is where you should test your plugin's on_complete method.
    Use alteryx_g_p_t_plugin_service.run_on_complete to run the plugin's on_complete method.

    Once the method has run, you can compare the output data against expected values,
    by accessing the corresponding data from alteryx_g_p_t_plugin_service.data_streams.
    Use the output anchor name as the dictionary key.
    If no data was written, alteryx_g_p_t_plugin_service.data_streams will be an empty dictionary.

    You can also compare IO calls made to designer through alteryx_g_p_t_plugin_service.io_stream.
    The message type (INFO, WARN, ERROR) will be prepended to the message's text with a colon.
    If no provider.io methods were called, alteryx_g_p_t_plugin_service.io_stream will be an empty list.
    """
    alteryx_g_p_t_plugin_service.run_on_complete()
    
    assert alteryx_g_p_t_plugin_service.data_streams == {}
    assert alteryx_g_p_t_plugin_service.io_stream == ["INFO:AlteryxGPT tool done."]

    