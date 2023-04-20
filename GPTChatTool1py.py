import AlteryxPythonSDK as Sdk
import openai
import re
from xml.dom import minidom
import requests
import time


class GPTChatTool:
    def __init__(self, n_tool_id: int, alteryx_engine: object, output_anchor_mgr: object):
        self.n_tool_id = n_tool_id
        self.alteryx_engine = alteryx_engine
        self.output_anchor_mgr = output_anchor_mgr
        self.output_anchor = None

    def pi_init(self, str_xml: str):
        self.output_anchor = self.output_anchor_mgr.get_output_anchor('Output')

    def pi_add_incoming_connection(self, str_type: str, str_name: str) -> object:
        return self

    def pi_add_outgoing_connection(self, str_name: str) -> bool:
        return True

    def pi_push_all_records(self, n_record_limit: int) -> bool:
        self.alteryx_engine.output_message(
            self.n_tool_id, Sdk.EngineMessageType.error, 'Missing incoming connection.')
        return False

    def pi_close(self, b_has_errors: bool):
        self.output_anchor.assert_close()

    def ii_init(self, record_info_in: object) -> bool:
        self.record_info_in = record_info_in
        self.record_info_out = record_info_in.clone()
        self.output_anchor.init(self.record_info_out)
        return True

    def ii_push_record(self, in_record: object) -> bool:
        # Get the input column data
        input_value = None
        for field in self.record_info_in:
            field_name = field.get_name()
            field_type = field.get_type()
            field_value = self.record_info_in.get_field_data(
                in_record, field_name)
            if field_type == Sdk.FieldType.string:
                input_value = field_value
            elif field_type == Sdk.FieldType.blob:
                input_value = field_value.decode('utf-8')

        if input_value is None:
            self.alteryx_engine.output_message(
                self.n_tool_id, Sdk.EngineMessageType.error, 'Input value is empty.')
            return False

        # Call the GPT function with the input value
        prompt = self.get_prompt(input_value)
        gpt_result = self.get_gpt_response(prompt)

        # Create a new output record
        out_record = self.record_info_out.construct_record(in_record)

        # Set the output column value with the GPT result
        output_column_index = self.record_info_out.add_field(
            'Output', Sdk.FieldType.string, 1073741823)
        self.record_info_out.set_field_data(
            out_record, output_column_index, gpt_result)

        # Push the output record to the output anchor
        self.output_anchor.push_record(out_record)

        return True

    def get_prompt(self, input_value):
        prompt_header = (
            "Please provide the following information:\n\n"
        )

        prompt_body = f"{input_value}\n\n"

        prompt_footer = (
            "If you don't know the answer, you can search it on Google and provide the information.\n"
        )

        prompt = prompt_header + prompt_body + prompt_footer
        return prompt


    def get_gpt_response(prompt, model, api_key, max_tokens, google_search_api_key=None, google_search_engine_id=None):
        # Initialize the OpenAI API client
        openai.api_key = api_key

        # Count the number of tokens in the prompt
        prompt_tokens = count_tokens(prompt, model)

        # Split the prompt into multiple parts if it exceeds the maximum number of tokens allowed
        prompt_parts = split_text(prompt, model, max_tokens - prompt_tokens)

        # Create an empty list to store the chat history
        chat_history = []

        # Loop through each prompt part and generate a GPT-3 response
        for part in prompt_parts:
            # Append the current prompt part to the chat history
            chat_history.append({"role": "user", "content": part})

            # Generate the GPT-3 response
            response = openai.Completion.create(
                engine=model,
                prompt=chat_history,
                max_tokens=max_tokens,
            )

            # Append the GPT-3 response to the chat history
            chat_history.append(
                {"role": "AI", "content": response.choices[0].text})

            # Check if the GPT-3 response contains any placeholders for missing information
            if '[MISSING INFO]' in response.choices[0].text:
                # Extract the missing information from the GPT-3 response using Google search
                missing_info = extract_missing_info(
                    response.choices[0].text, google_search_api_key, google_search_engine_id)

                # Replace the placeholders in the GPT-3 response with the extracted missing information
                response.choices[0].text = response.choices[0].text.replace(
                    '[MISSING INFO]', missing_info)

        # Return the final GPT-3 response
        return chat_history[-1]['content']


    def ii_push_record(self, in_record: object, input_column_name: str, output_column_name: str, prompt_column_name: str, api_key: str, model: str, google_search: bool, google_cse_id: str, google_api_key: str) -> bool:
        # Get the input column data
        input_column_index = self.record_info_in.get_field_num(input_column_name)
        input_value = self.record_info_in.get_field_data(
            in_record, input_column_index, Sdk.FieldReturnType.string)

        # Get the prompt column data
        prompt_column_index = self.record_info_in.get_field_num(prompt_column_name)
        prompt_value = self.record_info_in.get_field_data(
            in_record, prompt_column_index, Sdk.FieldReturnType.string)

        # Call the GPT function with the input value
        gpt_result = get_gpt_response(
            prompt_value, input_value, api_key, model, google_search, google_cse_id, google_api_key)

        # Create a new output record
        out_record = self.record_info_out.construct_record(in_record)

        # Set the output column value with the GPT result
        output_column_index = self.record_info_out.get_field_num(
            output_column_name)
        self.record_info_out.set_field_data(
            out_record, output_column_index, gpt_result)

        # Push the output record to the output anchor
        self.output_anchor.push_record(out_record)

        return True

    def extract_requirements(response):
        # Split the response into lines
        lines = response.split('\n')

        # Define a regular expression pattern to match application name, CPU count, and RAM count
        pattern = r"Application '(?P<app_name>.*?)': [cpus](?P<cpu_count>.*?)[/cpus], [ram](?P<ram_count>.*?)[/ram]$"

        # Create a dictionary to store the CPU count and RAM count for each application
        requirements_dict = {}

        for line in lines:
            match = re.match(pattern, line)
            if match:
                app_name = match.group('app_name')
                cpu_count = match.group('cpu_count')
                ram_count = match.group('ram_count')
                requirements_dict[app_name] = {
                    "cpu_count": cpu_count, "ram_count": ram_count}

        return requirements_dict

    def create_dataframe(requirements_dict):
        # Create lists to store the data for each column in the dataframe
        applications = []
        cpu_counts = []
        ram_counts = []

        # Iterate through the dictionary and append the data to the lists
        for app_name, requirements in requirements_dict.items():
            applications.append(app_name)
            cpu_counts.append(requirements["cpu_count"])
            ram_counts.append(requirements["ram_count"])

        # Create a new dataframe with the data
        data = {"Application": applications,
                "Minimum CPU count": cpu_counts, "Minimum RAM count": ram_counts}
        df = pd.DataFrame(data)

        return df


    def save_to_csv(df, output_file):
        # Save the dataframe to a CSV file
        df.to_csv(output_file, index=False)

        print(f"Saved to {output_file}")
        def pi_init(self, str_xml: str):
        self.output_anchor = self.output_anchor_mgr.get_output_anchor('Output')

        # Parse the incoming XML configuration
        xml_doc = minidom.parseString(str_xml)
        configuration = xml_doc.getElementsByTagName("Configuration")[0]
        self.input_column_name = configuration.getElementsByTagName("InputColumn")[0].firstChild.data.strip()
        self.output_column_name = configuration.getElementsByTagName("OutputColumn")[0].firstChild.data.strip()
        self.prompt_column_name = configuration.getElementsByTagName("PromptColumn")[0].firstChild.data.strip()
        self.prompt_length = int(configuration.getElementsByTagName("PromptLength")[0].firstChild.data.strip())
        self.engine = configuration.getElementsByTagName("Engine")[0].firstChild.data.strip()
        self.model = configuration.getElementsByTagName("Model")[0].firstChild.data.strip()
        self.api_key = configuration.getElementsByTagName("APIKey")[0].firstChild.data.strip()
        self.google_key = configuration.getElementsByTagName("GoogleKey")[0].firstChild.data.strip()
        self.google_cx = configuration.getElementsByTagName("GoogleCX")[0].firstChild.data.strip()

        # Initialize the OpenAI API with the user's API key
        openai.api_key = self.api_key


    def ii_push_record(self, in_record: object, prompt_column: str, input_column: Optional[str], output_column: str) -> bool:
        # Get the input value(s)
        if input_column:
            input_column_index = self.record_info_in.get_field_num(input_column)
            input_value = self.record_info_in.get_field_data(
                in_record, input_column_index, Sdk.FieldReturnType.string)
        else:
            input_value = None

        # Get the prompt value
        prompt_column_index = self.record_info_in.get_field_num(prompt_column)
        prompt_value = self.record_info_in.get_field_data(
            in_record, prompt_column_index, Sdk.FieldReturnType.string)

        # Call the get_gpt_response function with the prompt and input values
        gpt_response = get_gpt_response(prompt_value, input_value)

        # Create a new output record
        out_record = self.record_info_out.construct_record(in_record)

        # Set the output column value with the GPT response
        output_column_index = self.record_info_out.get_field_num(output_column)
        self.record_info_out.set_field_data(
            out_record, output_column_index, gpt_response)

        # Push the output record to the output anchor
        self.output_anchor.push_record(out_record)

        return True
