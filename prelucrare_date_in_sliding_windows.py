import json
import re

input_path = "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/date_eLife_simplificate_prelucrate.json"
output_path = "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/date_eLife_simplificate_prelucrate_sliding.json"

with open(input_path, "r") as infile:
    data = json.load(infile)


def split_into_words(text):
    return re.findall(r'\b\w+\b|[.,!?;:"\'()]', text)


def generate_windows(token_list, window_size, step):
    windows = []
    while len(token_list) > 0:
        window = token_list[:window_size]
        # print(len(window))
        if len(window) < window_size:
            windows.append(window)
            break
        windows.append(window)
        token_list = token_list[step:]

    return windows


window_size = 512
step = 100
extended_json = []


for item in data:
    text = item["sections"]
    words = split_into_words(text)
    windows = generate_windows(words, window_size, step)

    extended_json.append(
        {
            "id": item["id"],
            "windows": [" ".join(window) for window in windows],
            "summary": item["summary"],
        }
    )


with open(output_path, "w") as outfile:
    json.dump(extended_json, outfile, indent=4)
