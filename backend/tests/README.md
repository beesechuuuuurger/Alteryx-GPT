# Testing basics with pytest

## What is pytest?

Pytest is a unit testing framework, meant to test python code.

## Installing pytest

You can install pytest by running this command:

`pip install pytest`

## Running pytest

Once you have pytest installed, you should be able to run your tests by simply running

`pytest .\backend`

Pytest will then look through this directory for any python functions that fit the 

### IDE integration

One nice thing about pytest is that it works well with all widely used integrated development environments - PyCharm, VSCode, etc.
You can discover and run tests from these IDEs with the touch of a button, and get test results delivered in a nice breakdown.
However, there may be some setup involved in getting this integration working:

#### PyCharm

Go to File > Settings > Tools > Integrated Python Tools, and make sure that under Testing, the default test runner is set to pytest.

Once this is set, PyCharm will take a moment to index your files and discover the tests. You can tell when it's done doing so when you open the test file, and you see a little green run button on the left of each of the tests you can run.
You can run all of the tests in a file by right-clicking on the file and clicking the `Run pytest in <filename>` option.

#### VSCode

When you open VSCode, you should see a Testing tab on the left-hand side (logo is a flask)
Click on this tab, and it should tell you that no tests have been found yet, and that you can configure a test framework.
Click on the "Configure Python Tests" button, pick `pytest`, use the `backend` folder as the directory containing the tests, the discovered tests should populate in that window.

#### Notes

If test discovery fails, and the error message given is an import error, this could be due to an environment issue.
Check the bottom-right corner of the IDE to make sure it's set to the conda environment that your `ayx_python_sdk` and `ayx_plugin_cli` are installed in. 

You can set breakpoints in both IDEs - in roughly the same way! Hover your mouse over the line numbers on the left, and you'll see a little red dot show up.
Click, and run the test in debug mode, and you'll be able to stop execution at that point, and examine the values of local variables at that point. 

## Writing tests

By default, pytest discovers tests by searching for files that match the patterns `test_*.py` or `*_test.py`.
Within these files, any functions that match the pattern `test_*()` are marked as tests.

Use `assert` statements to check your code's values against an expected outcome - ie:

```
assert "value" in ["expected", "value"]
```

When a test is run, there are roughly three possible outcomes: 

a) The test code runs to the end without issue 
b) An unexpected error is raised 
c) the test fails on an assertion check

The goal of unit testing is to discover those test failures ahead of time, and debug your logic until your output matches the expected output. 

## Interacting with the plugin service

SdkToolTestService is a middleware layer that mocks out some of Designer's functionality, and allows you to test your plugin's callbacks in an isolated environment.
By default, we generate a pytest fixture that wraps and returns an instance of SdkToolTestService.
The class contains several helper methods and attributes to make your testing experience easier:

### io_stream

This attribute mocks out Designer's messages window - basically any strings that get sent over through `provider.io` calls.
When you run your test, you can examine `plugin_service_fixture.io_stream` to see which messages were captured, and compare them against a list of expected messages.
Note that `io_stream` is a list of strings, prepended with the `provider.io` call in question. This is roughly the format they follow:

`"<INFO|WARN|ERROR>:<message>"`

For example, this `provider.io` call in the plugin code:

`self.provider.io.info("Test Code")`

would show up as `"INFO:Test Code"` in `io_stream`. 

### data_streams

This attribute mocks out the plugin's output anchor - ie, any data that would show up in a Browse tool placed after the plugin in Designer, should show up here.
In the plugin code, this would be any data that is written to `provider.io.write_to_anchor()`. 
When you run your test, you can examine `plugin_service_fixture.data_streams` to ensure that the correct output data was written to the output anchor,
and compare the captured record batches against a list of expected record batches.   

For the purpose of simplicity, the completed stream is represented by a dictionary, typed like so:
```
{
    "<Output Anchor name>": [pa.RecordBatch]
}
```

if `provider.io.write_to_anchor` is never called, the `data streams` attribute should be an empty dictionary.

### run_on_record_batch

This method runs your plugin's on_record_batch method - pass it an input anchor and a corresponding record batch, and it should run the method and capture the I/O and data stream outputs

### run_on_incoming_connection_complete

This method runs your plugin's on_incoming_connection_complete method, on the specified input anchor, and captures the data and I/O output

### run_on_complete

This method runs your plugin's on_complete method, and captures the data and I/O output

# Autogenerated tests

By default, we generate these four tests:

```
test_init
test_on_record_batch
test_on_incoming_connection_complete
test_on_complete
```

But users can add as many, or as few, as needed. By default, these run the corresponding SdkToolTestService method, and compare them to default output.
One thing to note is that the on_record_batch test is parametrized, and will run three times by default, one for each batch named in the list argument.
These batches are defined in `conftest.py`, and should be edited, renamed, and changed to suit your testing needs. 


# Examples:

```
@pytest.mark.parametrize("record_batch", ["small_batch", "medium_batch", "large_batch"])
def test_on_record_batch(plugin_service_fixture, anchor, record_batch, request):
    record_batch = request.getfixturevalue(record_batch)
    plugin_service_fixture.run_on_record_batch(record_batch, anchor)
    #  In this case, since the tool is a simple passthrough, the input data should match the output data, 1-1.
    assert plugin_service_fixture.data_streams["Output"] == [record_batch]
    #  In this case, there are no calls being made to provider.io, so the io_stream for on_record_batch should be empty.
    assert plugin_service_fixture.io_stream == []
```

```
def test_on_incoming_connection_complete(plugin_service_fixture, anchor):
    plugin_service_fixture.run_on_incoming_connection_complete(anchor)
    #  In this case, no data was written to any of the output anchors, so the streams should be empty.
    assert plugin_service_fixture.data_streams == {}
    #  In this case, the only call being made is "Connection connection on Anchor anchor" as an info message.
    assert plugin_service_fixture.io_stream == [f"INFO:Connection {anchor.connection} on Anchor {anchor.name}"]
```

```
def test_on_complete(plugin_service_fixture):
    plugin_service_fixture.run_on_complete()
    #  In this case, no data was written to any of the output anchors, so the streams should be empty.
    assert plugin_service_fixture.data_streams == {}
    #  In this case, the only call being made is "Pass through tool done" as an info message.
    assert plugin_service_fixture.io_stream == ["INFO:Pass through tool done"]
```