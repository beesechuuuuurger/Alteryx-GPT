import openai
import pandas as pd
import tiktoken
import re
import time
import csv
from autogpt.pagination import paginate,
from autogpt.utils import load_api_key, save_memory_to_disk, load_memory_from_disk, google_search

# Load the CSV files
applications_df = pd.read_csv(
    "D:\\DEV\\app dependencies\\min required\\min required input.csv")

# Create a list of applications
applications = applications_df['application'].tolist()

# Define your OpenAI API key
openai.api_key = "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"
model = "gpt-3.5-turbo"
encoding = tiktoken.encoding_for_model(model)
MAX_TOKENS = 4096

memory_filename = "memory.json"
memory = load_memory_from_disk(memory_filename)

# Function to check if application information is already in memory


def check_memory(app_name):
    for item in memory:
        if item['application'] == app_name:
            return True, item['cpus'], item['ram']
    return False, None, None

# Function to save application information to memory


def save_to_memory(app_name, cpus, ram):
    memory.append({'application': app_name, 'cpus': cpus, 'ram': ram})
    save_memory_to_disk(memory, memory_filename)


def count_tokens(texts):
    token_count = 0
    try:
        if isinstance(texts, str):
            token_count += len(list(encoding.encode(texts)))
        elif isinstance(texts, list):
            token_count += len(texts)
    except KeyError:
        print("Warning: model not found. Using default encoding.")
        pass
    print(token_count)
    return token_count



def split_text(text, max_tokens, token_count=0):
    safety_margin = 1500  # A margin to keep the token count safely within the limit
    result = []
    current_chunk = ""
    current_chunk_tokens = []

    for token in encoding.encode(text):
        token_str = encoding.decode([token])
        if count_tokens(current_chunk_tokens + [token]) <= max_tokens - safety_margin:
            current_chunk_tokens.append(token)
            current_chunk += token_str
        else:
            result.append(current_chunk.strip())
            current_chunk_tokens = [token]
            current_chunk = token_str

    if current_chunk:
        result.append(current_chunk.strip())

    print(result)
    return result


def generate_prompt(apps, memory):
    prompt_header = (
        "Please provide the minimum hardware requirements, "
        "specifically CPU count and RAM size, "
        "for the following applications, "
        "assuming typical usage scenarios and default configurations. "
        "If possible, also consider the most commonly used version of each application.\n\n"
    )

    memory_prompt = "Here's what I remember about the applications:\n"
    for item in memory:
        memory_prompt += (
            f"{item['application']}: [cpus]{item['cpus']}[/cpus], [ram]{item['ram']}[/ram]\n"
        )

    command_prompt = (
        "\nIf you don't know the answer, you can search it on Google and provide the information.\n"
    )

    list_prompt = "List of applications:\n\n"
    for app in apps:
        list_prompt += f"{app}: [cpus]{{}}[/cpus], [ram]{{}}[/ram]\n"

    prompt = prompt_header + memory_prompt + command_prompt + list_prompt
    return prompt




def get_response(prompt):
    tokens = count_tokens(prompt)
    
    if tokens > MAX_TOKENS:
        print("Warning: Prompt too long. Skipping.")
        return ""

    chat_history = []
    chat_history.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        max_tokens=2500,
    )

    generated_text = response['choices'][0]['message']['content'].strip()
    print(generated_text)
    return generated_text




def extract_requirements(output):
    app_names = []
    cpu_counts = []
    ram_counts = []
    
    # Split the output into lines
    lines = output.split('\n')
    
    # Define a regular expression pattern to match application name, CPU count, and RAM count
    pattern = r"Application '(?P<app_name>.*?)': [cpus](?P<cpu_count>.*?)[/cpus], [ram](?P<ram_count>.*?)[/ram]$"

    for line in lines:
        match = re.match(pattern, line)
        if match:
            app_name = match.group('app_name')
            cpu_count = match.group('cpu_count')
            ram_count = match.group('ram_count')
            app_names.append(app_name)
            cpu_counts.append(cpu_count)
            ram_counts.append(ram_count)
    print(app_names, cpu_counts, ram_counts)
    return app_names, cpu_counts, ram_counts




def create_dataframe(applications, cpu_counts, ram_counts):
    data = {'Application': applications, 'Minimum CPU count': cpu_counts, 'Minimum RAM count': ram_counts}
    df = pd.DataFrame(data)
    print(df)
    return df

def save_to_csv(applications, cpu_counts, ram_counts, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Application', 'Minimum CPU count', 'Minimum RAM count'])
        for app, cpu_count, ram_count in zip(applications, cpu_counts, ram_counts):
            csv_writer.writerow([app, cpu_count, ram_count])


unknown_apps = []
known_apps = []
known_cpus = []
known_rams = []

# Check memory for known applications
for app in applications:
    is_known, cpus, ram = check_memory(app)
    if is_known:
        known_apps.append(app)
        known_cpus.append(cpus)
        known_rams.append(ram)
    else:
        unknown_apps.append(app)

# Get requirements for unknown applications
if unknown_apps:
    prompt = generate_prompt(unknown_apps, memory)
    response = get_response(prompt)
    app_names, cpu_counts, ram_counts = extract_requirements(response)

    # Save unknown applications information to memory
    for app_name, cpu_count, ram_count in zip(app_names, cpu_counts, ram_counts):
        save_to_memory(app_name, cpu_count, ram_count)

    # Combine known and unknown applications information
    all_apps = known_apps + app_names
    all_cpus = known_cpus + cpu_counts
    all_rams = known_rams + ram_counts
else:
    all_apps = known_apps
    all_cpus = known_cpus
    all_rams = known_rams
    print(all_apps)
    print(all_cpus)
    print(all_rams)
# Create and save the dataframe to CSV
df = create_dataframe(all_apps, all_cpus, all_rams)
save_to_csv(all_apps, all_cpus, all_rams,'D:\\DEV\\app dependencies\\min required\\min required output.csv')
print(f"Saved to CSV")
