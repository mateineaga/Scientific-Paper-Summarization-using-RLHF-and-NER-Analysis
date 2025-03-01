import json

# Read the input data from text.json
with open(
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/date_eLife_simplificate.json",
    "r",
) as file:
    data = json.load(file)

# Transform data into the desired format
transformed_data = []
for item in data:
    # Merge all sections into a single list of strings
    merged_sentence = [" ".join(sentence) for sentence in item["sections"]]
    merged_sections = " ".join(merged_sentence)

    merged_summary = " ".join(item["summary"])
    # merged_summaries = " ".join(merged_summary)

    # Create a new dictionary with the required fields
    transformed_item = {
        "id": item["id"],
        "sections": merged_sections,
        "summary": merged_summary,
    }

    # Add to the transformed data list
    transformed_data.append(transformed_item)

# Write the transformed data to a new JSON file
with open(
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/date_eLife_simplificate_prelucrate.json",
    "w",
) as outfile:
    json.dump(transformed_data, outfile, indent=4)
