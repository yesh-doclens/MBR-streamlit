import pdfplumber
import json
import re
import sys
import os


def extract_and_eval_json(text: str):
    """
    Extracts a substring that starts with { and ends with } from the given text
    and evaluates it as JSON.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return eval(match.group())
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return {}
    return {}


def get_output_from_sonnet(bedrock_client, prompt):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    response = bedrock_client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    response_text = response_body["content"][0]["text"]
    return response_text


necessity_prompt = (
    lambda medical_summary_output, claim_summary_output, bill_items: f"""
You are an expert medical bill reviewer. You will be given two summarized data. One is medical summary output which has the conditions categorized by "Cause", "Accident", "Pre-existing Condition", and "Comorbidity". The other is claim summary output which has the injuries/medical conditions identified, injuries/medical conditions related to accident or incident and confirmed by the medical providers, and medical conditions or injuries prior to accident or not related to accident.
Then you will be given all the medical procedures and drug items billed in the medical bill. Now, your task is to associate the necessity of each of the bill items and categorize strictly into the following three categories:
- "Necessary and Medically stated": The bill item is necessary (accident related) and also confirmed by the medical providers in the claim summary output.
- "Necessary but Not Medically stated": The bill item is necessary (accident related) but not yet confirmed by the medical providers in the claim summary output.
- "Not necessary": The bill item is not necessary (not accident related). It might be pre-existing conditions or comorbidity which have not been affected after the accident.

The output should strictly be a JSON output with the bill items as keys and the values as a json containing the category key which is one of the above 3, and the reason key which is a very brief reason explaining the categorization.
Here are the input information:
Medical Summary Output: {medical_summary_output}
Claim Summary Output: {claim_summary_output}
Bill Items: {bill_items}

Output:
"""
)


def get_necessity_json(
    bedrock_client, medical_summary_output, claim_summary_output, bill_items
):
    prompt = necessity_prompt(medical_summary_output, claim_summary_output, bill_items)
    response_text = get_output_from_sonnet(bedrock_client, prompt)
    try:
        response_text = get_output_from_sonnet(bedrock_client, prompt)
        json_output = extract_and_eval_json(response_text)
        return json_output
    except Exception as e:
        print("Error decoding JSON from model response:")
        print(response_text)
        raise e


def clean_text(text):
    """Remove citation markers like [1][2][3] from text."""
    cleaned = re.sub(r"$$\d+$$", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_list_items(text_block):
    """Extract numbered list items from a text block."""
    lines = text_block.split("\n")
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match lines starting with a number and period (e.g., "1.", "2.")
        match = re.match(r"^\d+\.\s+(.+)", line)
        if match:
            items.append(clean_text(match.group(1)))
    return items


def extract_damages_injuries_from_claim_summary(pdf_path):
    """
    Extract specific fields from the Damages/Injuries section of a PDF.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        dict: Extracted data as a dictionary.
    """
    full_text = ""

    # Extract all text from the PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    # -------------------------------------------------------------------------
    # Locate the Damages/Injuries section
    # -------------------------------------------------------------------------
    damages_pattern = re.search(
        r"3\.\s*Damages/Injuries(.*?)(?=4\.\s*Economic Damages Claimed|$)",
        full_text,
        re.DOTALL | re.IGNORECASE,
    )

    if not damages_pattern:
        print("Warning: 'Damages/Injuries' section not found in the PDF.")
        damages_text = full_text  # Fall back to full text
    else:
        damages_text = damages_pattern.group(1)

    # -------------------------------------------------------------------------
    # 1. Injuries/Medical Conditions Identified
    # -------------------------------------------------------------------------
    injuries_identified = []

    injuries_identified_match = re.search(
        r"Injuries/Medical Conditions Identified\s*(.*?)"
        r"(?=Injuries/Medical conditions Related to Accident|$)",
        damages_text,
        re.DOTALL | re.IGNORECASE,
    )

    if injuries_identified_match:
        block = injuries_identified_match.group(1)
        items = extract_list_items(block)
        if items:
            injuries_identified = items
        else:
            # If no numbered list found, capture the raw cleaned text
            injuries_identified = [clean_text(block)]

    # -------------------------------------------------------------------------
    # 2. Injuries/Medical Conditions Related to Accident or Incident
    # -------------------------------------------------------------------------
    injuries_related = ""

    injuries_related_match = re.search(
        r"Injuries/Medical conditions Related to Accident or Incident\s*(.*?)"
        r"(?=Medical conditions or injuries that existed prior|$)",
        damages_text,
        re.DOTALL | re.IGNORECASE,
    )

    if injuries_related_match:
        block = injuries_related_match.group(1)
        injuries_related = clean_text(block)

    # -------------------------------------------------------------------------
    # 3. Medical Conditions or Injuries Prior to Accident / Not Related
    # -------------------------------------------------------------------------
    pre_existing = []

    pre_existing_match = re.search(
        r"Medical conditions or injuries that existed prior to the accident or incident"
        r".*?accident/incident\s*(.*?)"
        r"(?=Medical providers that have examined|$)",
        damages_text,
        re.DOTALL | re.IGNORECASE,
    )

    if pre_existing_match:
        block = pre_existing_match.group(1)
        items = extract_list_items(block)
        if items:
            pre_existing = items
        else:
            pre_existing = [clean_text(block)]

    # -------------------------------------------------------------------------
    # Build output dictionary
    # -------------------------------------------------------------------------
    output = {
        "Injuries/Medical Conditions Identified": injuries_identified,
        "Injuries/Medical Conditions Related to Accident or Incident and Confirmed by Medical Providers": injuries_related,
        "Medical conditions or injuries that existed prior to the accident or incident "
        "or are not related to the accident/incident": pre_existing,
    }

    return output


def extract_medical_conditions_from_medical_summary(medical_entities):
    """
    Extract and categorize medical conditions from the medical summary entities.

    Args:
        medical_entities: List of medical entities extracted from the summary,
                        each with 'text' and 'label' (e.g., 'Condition', 'Medication')
    """
    final_json = {}
    final_json["Cause"] = []
    final_json["Accident"] = []
    final_json["Pre-existing Condition"] = []
    final_json["Comorbidity"] = []
    patients_data = medical_entities.get("patients", [])
    for patient_data in patients_data:
        medical_entities = patient_data.get("medical_history", [])
        for medical_condition in medical_entities:
            condition_name = medical_condition.get("condition_type", "jklkjl")
            if not condition_name:
                continue
            condition_string = f"{medical_condition.get('icd10cm_concept_code', '')} - {medical_condition.get('answer', '')}"
            if "Cause" in condition_name:
                final_json["Cause"].append(condition_string)
            elif "Accident" in condition_name:
                final_json["Accident"].append(condition_string)
            elif "Pre-existing Condition" in condition_name:
                final_json["Pre-existing Condition"].append(condition_string)
            elif "Comorbidity" in condition_name:
                final_json["Comorbidity"].append(condition_string)

    return final_json


def main():
    pdf_path = "./fake_files/GP-File-2-claim_summary.pdf"  # Override for testing without command-line args
    base_name = os.path.splitext(pdf_path)[0]
    output_json_path = base_name + "_damages_injuries.json"
    print(f"Processing: {pdf_path}")
    data = extract_damages_injuries_from_claim_summary(pdf_path)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Output saved to: {output_json_path}")
    print("\nExtracted Data:")
    print(json.dumps(data, indent=4))


if __name__ == "__main__":
    main()
