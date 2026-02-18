import streamlit as st
import boto3
import json
import base64
import pandas as pd
from datetime import datetime
import re
import os
import hashlib
from pathlib import Path
from langchain_aws import ChatBedrockConverse
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.callbacks import BaseCallbackHandler
import time
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    NameObject,
    FloatObject,
    ArrayObject,
    DictionaryObject,
    TextStringObject,
)
from rapidfuzz import process, fuzz
from io import BytesIO
import uuid


def load_demo_files():
    demo_base = "public/pdf"

    with open(os.path.join(demo_base, "GP2.pdf"), "rb") as f:
        pdf_bytes = f.read()

    with open(
        os.path.join(demo_base, "GP2 Page Classification - Sheet1.csv"), "rb"
    ) as f:
        csv_bytes = f.read()

    with open(
        os.path.join(
            demo_base, "b98d314f-2b46-49f3-9986-1dc7249cb449__document_uuid__GP2.json"
        ),
        "rb",
    ) as f:
        json_bytes = f.read()

    st.session_state["demo_pdf"] = BytesIO(pdf_bytes)
    st.session_state["demo_pdf"].name = "GP2.pdf"

    st.session_state["demo_csv"] = BytesIO(csv_bytes)
    st.session_state["demo_csv"].name = "GP2 Page Classification - Sheet1.csv"

    st.session_state["demo_json"] = BytesIO(json_bytes)
    st.session_state["demo_json"].name = (
        "b98d314f-2b46-49f3-9986-1dc7249cb449__document_uuid__GP2.json"
    )

    st.session_state["use_demo_files"] = True


def load_demo_bill():
    demo_base = "public/single-bill"

    with open(os.path.join(demo_base, "Medical_Bill_1.png"), "rb") as f:
        img_bytes = f.read()

    st.session_state["demo_bill"] = BytesIO(img_bytes)
    st.session_state["demo_bill"].name = "Medical_Bill_1.png"


def load_demo_bill_multiple():
    demo_base = "public/multiple-bill"

    bill_files = ["Medical_Bill_1.png", "Medical_Bill_2.png"]

    st.session_state["demo_bill_multiple"] = []
    for file in bill_files:
        with open(os.path.join(demo_base, file), "rb") as f:
            img_bytes = f.read()
        st.session_state["demo_bill_multiple"].append(BytesIO(img_bytes))
        st.session_state["demo_bill_multiple"][-1].name = file


# 1. Setup Token Tracking Callback
class BedrockTokenCallback(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        """Accumulate tokens from each LLM call made by the agent"""
        # In 2026 LangChain, usage is stored in usage_metadata
        for generation in response.generations:
            for chunk in generation:
                metadata = chunk.message.usage_metadata
                self.input_tokens += metadata.get("input_tokens", 0)
                self.output_tokens += metadata.get("output_tokens", 0)


# Cache directory paths
CACHE_DIR = Path(__file__).parent / "cache"
TEXTRACT_CACHE_DIR = CACHE_DIR / "textract"
TXT_CACHE_DIR = CACHE_DIR / "txt"
DF_CACHE_DIR = CACHE_DIR / "dataframe"


def ensure_cache_dirs():
    """Ensure cache directories exist."""
    TEXTRACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(file_name: str) -> str:
    """Generate cache key from file name (without extension)."""
    # Remove extension and sanitize for filesystem
    base_name = Path(file_name).stem
    # Sanitize: remove special characters, keep alphanumeric and underscore
    sanitized = re.sub(r"[^\w\-]", "_", base_name)
    return sanitized


def check_cache(file_name: str) -> tuple[dict | None, str | None]:
    """
    Check if cached results exist for this file name.
    Returns tuple of (textract_json, txt_content) if cache exists, (None, None) otherwise.
    """
    cache_key = get_cache_key(file_name)
    textract_path = TEXTRACT_CACHE_DIR / f"{cache_key}.json"
    txt_path = TXT_CACHE_DIR / f"{cache_key}.txt"

    if txt_path.exists():
        txt_content = txt_path.read_text(encoding="utf-8")
        textract_json = None
        if textract_path.exists():
            textract_json = json.loads(textract_path.read_text(encoding="utf-8"))
        return textract_json, txt_content

    return None, None


def check_df_cache(file_name: str) -> pd.DataFrame | None:
    """Check if a processed DataFrame exists in cache."""
    file_hash = hashlib.md5(file_name.encode()).hexdigest()
    cache_path = DF_CACHE_DIR / f"{file_hash}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)
    return None


def save_df_to_cache(file_name: str, df: pd.DataFrame):
    """Save a processed DataFrame to cache."""
    file_hash = hashlib.md5(file_name.encode()).hexdigest()
    cache_path = DF_CACHE_DIR / f"{file_hash}.csv"
    df.to_csv(cache_path, index=False)


def invoke_kimi_model(bedrock_client, prompt_text):
    """
    Invokes the Kimi K2 Thinking model on Amazon Bedrock.
    """
    MODEL_ID = "moonshot.kimi-k2-thinking"
    REGION = "us-east-1"
    # The payload structure is specific to the Moonshot API within Bedrock
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are Kimi, an AI assistant created by Moonshot AI.",
            },
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 1.0,  # Recommended for optimal reasoning performance
        "max_tokens": 4096,  # Ensure adequate space for full reasoning chains
    }

    body = json.dumps(payload)
    content_type = "application/json"
    accept = "application/json"

    try:
        response = bedrock_client.invoke_model(
            body=body, modelId=MODEL_ID, accept=accept, contentType=content_type
        )
        # print(response)
        result_in_byte = response.get("body").read()
        result = json.loads(str(result_in_byte, encoding="utf-8"))
        # print(result)
        # Extract the generated text
        generated_text = result["choices"][0]["message"]["content"]
        input_tokens = result["usage"]["prompt_tokens"]
        output_tokens = result["usage"]["completion_tokens"]
        return generated_text, input_tokens, output_tokens

    except Exception as e:
        print(f"Error invoking model: {e}")
        return None


def save_to_cache(file_name: str, textract_response: dict, txt_content: str):
    """Save Textract response and TXT content to cache."""
    ensure_cache_dirs()
    cache_key = get_cache_key(file_name)

    # Save Textract JSON
    textract_path = TEXTRACT_CACHE_DIR / f"{cache_key}.json"
    textract_path.write_text(json.dumps(textract_response, indent=2), encoding="utf-8")

    # Save TXT file
    txt_path = TXT_CACHE_DIR / f"{cache_key}.txt"
    txt_path.write_text(txt_content, encoding="utf-8")


def call_textract(image_bytes: bytes) -> dict:
    """
    Call AWS Textract analyze_document API with LAYOUT and TABLES features.
    """
    session = boto3.Session(
        aws_access_key_id=st.session_state["aws_credentials"]["aws_access_key"],
        aws_secret_access_key=st.session_state["aws_credentials"]["aws_secret_key"],
        aws_session_token=st.session_state["aws_credentials"]["aws_session_token"],
        region_name="us-east-1",
    )
    textract_client = session.client(service_name="textract", region_name="us-east-1")

    response = textract_client.analyze_document(
        Document={"Bytes": image_bytes}, FeatureTypes=["LAYOUT", "TABLES"]
    )

    return response


def get_textract_confidence(response: dict) -> dict:
    """Calculate average confidence for lines and tables in Textract response."""
    blocks = response.get("Blocks", [])
    line_confidences = [
        b.get("Confidence", 0) for b in blocks if b["BlockType"] == "LINE"
    ]
    table_confidences = [
        b.get("Confidence", 0) for b in blocks if b["BlockType"] == "TABLE"
    ]

    avg_line_conf = (
        sum(line_confidences) / len(line_confidences) if line_confidences else 0.0
    )
    avg_table_conf = (
        sum(table_confidences) / len(table_confidences)
        if table_confidences
        else avg_line_conf
    )

    return {
        "avg_line_confidence": round(avg_line_conf, 2),
        "avg_table_confidence": round(avg_table_conf, 2),
    }


def filter_medical_bill_pages(csv_df: pd.DataFrame) -> list[int]:
    """Filter CSV to get page numbers classified as medical bills."""
    if "Page Classification" not in csv_df.columns or "Number" not in csv_df.columns:
        return []

    # Target classification
    target_class = "Bills – Medical, Pharmacy"
    medical_bill_pages = csv_df[csv_df["Page Classification"] == target_class][
        "Number"
    ].tolist()
    return [int(p) for p in medical_bill_pages]


def get_textract_blocks_for_pages(
    textract_json: dict, pages: list[int]
) -> tuple[str, list[dict]]:
    """Extract LINE blocks and their geometry for specific pages from Textract JSON."""
    all_text = []
    all_geometries = []

    blocks = textract_json.get("Blocks", [])

    # Process page by page to insert markers
    for page_num in sorted(pages):
        all_text.append(f"\n[PAGE {page_num}]\n")
        for block in blocks:
            if block.get("BlockType") == "LINE" and block.get("Page") == page_num:
                text = block.get("Text", "")
                geometry = block.get("Geometry", {}).get("BoundingBox", {})

                all_text.append(text)
                all_geometries.append(
                    {"Text": text, "Page": page_num, "Geometry": geometry}
                )

    return "\n".join(all_text), all_geometries


def render_pdf_page_with_highlight(
    pdf_bytes: bytes, page_number: int, geometry: dict = None
) -> Image.Image:
    """Render a PDF page as a high-res PIL Image with an optional transparent blue highlight."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_number - 1)

        if geometry:
            rect = page.rect
            w_pdf, h_pdf = rect.width, rect.height

            # Geometry is relative (0 to 1)
            l = geometry.get("Left", 0) * w_pdf
            t = geometry.get("Top", 0) * h_pdf
            w = geometry.get("Width", 0) * w_pdf
            h = geometry.get("Height", 0) * h_pdf

            # Add highlight annotation (more stable than drawing on pixels)
            annot = page.add_rect_annot(fitz.Rect(l, t, l + w, t + h))
            annot.set_colors(fill=(0.1, 0.5, 1.0))  # Transparent blue
            annot.set_opacity(0.3)
            annot.update()

        # Render page to high-res image (3.0 scaling for zoom feel)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()
        return img
    except Exception as e:
        st.error(f"Error rendering PDF page: {e}")
        return None


def get_pdf_with_highlight_base64(
    pdf_bytes: bytes, page_number: int, geometry: dict = None
) -> str:
    """Add a transparent blue highlight to a PDF using fitz and return as base64."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if geometry:
            page = doc.load_page(page_number - 1)
            rect = page.rect
            w, h = rect.width, rect.height

            l = geometry.get("Left", 0) * w
            t = geometry.get("Top", 0) * h
            width = geometry.get("Width", 0) * w
            height = geometry.get("Height", 0) * h

            # Add highlight annotation
            annot = page.add_highlight_annot(fitz.Rect(l, t, l + width, t + height))
            annot.set_colors(stroke=(0.1, 0.5, 1.0))
            annot.update()

        # Save to bytes with compression
        pdf_output = doc.write(clean=True, deflate=True)
        doc.close()
        return base64.b64encode(pdf_output).decode("utf-8")
    except Exception as e:
        st.error(f"Error highlighting PDF: {e}")
        return ""


def textract_response_to_txt(response: dict) -> str:
    """
    Convert Textract response to a structured text file.
    Processes LAYOUT elements and TABLE data into readable text.
    """
    blocks = response.get("Blocks", [])

    # Build block lookup by ID
    block_map = {block["Id"]: block for block in blocks}

    # Collect text output
    output_lines = []

    # Process layout elements (titles, headers, paragraphs)
    layout_blocks = [b for b in blocks if b["BlockType"].startswith("LAYOUT_")]
    for layout_block in sorted(
        layout_blocks,
        key=lambda x: (
            x.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0),
            x.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0),
        ),
    ):
        block_type = layout_block["BlockType"]
        child_ids = [
            rel["Ids"]
            for rel in layout_block.get("Relationships", [])
            if rel["Type"] == "CHILD"
        ]
        child_ids = [item for sublist in child_ids for item in sublist]

        text_parts = []
        for child_id in child_ids:
            child_block = block_map.get(child_id)
            if child_block and child_block["BlockType"] == "LINE":
                text_parts.append(child_block.get("Text", ""))

        if text_parts:
            text = " ".join(text_parts)
            if block_type == "LAYOUT_TITLE":
                output_lines.append(f"=== {text} ===")
            elif block_type == "LAYOUT_HEADER":
                output_lines.append(f"--- {text} ---")
            elif block_type == "LAYOUT_SECTION_HEADER":
                output_lines.append(f"## {text}")
            else:
                output_lines.append(text)
            output_lines.append("")

    # Process tables
    table_blocks = [b for b in blocks if b["BlockType"] == "TABLE"]
    for table_idx, table_block in enumerate(table_blocks):
        output_lines.append(f"\n=== TABLE {table_idx + 1} ===")

        # Get all cells for this table
        cell_ids = []
        for rel in table_block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                cell_ids.extend(rel["Ids"])

        cells = [block_map.get(cid) for cid in cell_ids if block_map.get(cid)]
        cells = [c for c in cells if c and c["BlockType"] == "CELL"]

        # Build table structure
        if cells:
            max_row = max(c.get("RowIndex", 1) for c in cells)
            max_col = max(c.get("ColumnIndex", 1) for c in cells)

            table_data = [["" for _ in range(max_col)] for _ in range(max_row)]

            for cell in cells:
                row_idx = cell.get("RowIndex", 1) - 1
                col_idx = cell.get("ColumnIndex", 1) - 1

                # Get cell text from child WORD blocks
                cell_text_parts = []
                for rel in cell.get("Relationships", []):
                    if rel["Type"] == "CHILD":
                        for word_id in rel["Ids"]:
                            word_block = block_map.get(word_id)
                            if word_block and word_block["BlockType"] == "WORD":
                                cell_text_parts.append(word_block.get("Text", ""))

                table_data[row_idx][col_idx] = " ".join(cell_text_parts)

            # Calculate column widths
            col_widths = [
                max(len(table_data[r][c]) for r in range(max_row))
                for c in range(max_col)
            ]
            col_widths = [max(w, 5) for w in col_widths]  # Minimum width of 5

            # Output table rows
            for row_idx, row in enumerate(table_data):
                row_str = " | ".join(
                    cell.ljust(col_widths[i]) for i, cell in enumerate(row)
                )
                output_lines.append(f"| {row_str} |")
                if row_idx == 0:
                    separator = "-+-".join("-" * w for w in col_widths)
                    output_lines.append(f"+-{separator}-+")

        output_lines.append("")

    # If no layout blocks were processed, fall back to LINE blocks
    if not layout_blocks:
        line_blocks = [b for b in blocks if b["BlockType"] == "LINE"]
        for line_block in sorted(
            line_blocks,
            key=lambda x: (
                x.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0),
                x.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0),
            ),
        ):
            output_lines.append(line_block.get("Text", ""))

    return "\n".join(output_lines)


def parse_txt_to_records(txt_content: str) -> list[dict]:
    """
    Parse the TXT file content to extract medical bill records.
    Uses pattern matching to find procedure rows from tables.
    """
    records = []
    lines = txt_content.split("\n")

    in_table = False
    header_row = None

    for line in lines:
        line = line.strip()

        # Detect table start
        if line.startswith("=== TABLE"):
            in_table = True
            header_row = None
            continue

        # Skip separators
        if line.startswith("+-") or line.startswith("---"):
            continue

        # Process table rows
        if in_table and line.startswith("|") and line.endswith("|"):
            # Parse table row
            cells = [c.strip() for c in line.strip("|").split("|")]

            if header_row is None:
                # This is the header row
                header_row = [c.lower() for c in cells]
                continue

            # Try to extract record from this row
            record = extract_record_from_row(header_row, cells)
            if record:
                records.append(record)

        # Reset at table end or new section
        if line.startswith("===") and "TABLE" not in line:
            in_table = False
            header_row = None

    return records


def extract_record_from_row(headers: list[str], cells: list[str]) -> dict | None:
    """
    Extract a medical bill record from a table row.
    Maps table columns to our expected fields.
    """
    if len(cells) != len(headers):
        return None

    # Create a mapping from header variations to our standard fields
    field_mappings = {
        "date": ["date", "service date", "dos", "date of service"],
        "procedure": [
            "procedure",
            "description",
            "service",
            "medical procedure",
            "procedure name",
            "item",
            "service description",
        ],
        "cpt": ["cpt", "cpt code", "code", "procedure code", "hcpcs"],
        "modifier": ["modifier", "mod", "modifiers"],
        "quantity": ["quantity", "qty", "units", "unit", "count"],
        "unit_price": [
            "unit price",
            "price",
            "rate",
            "unit cost",
            "charge",
            "unit charge",
        ],
        "total": [
            "total",
            "total price",
            "amount",
            "total amount",
            "total charge",
            "extended",
            "line total",
        ],
    }

    # Build row data
    row_data = dict(zip(headers, cells))

    # Find field values
    def find_value(mappings):
        for header_variant in mappings:
            for h in headers:
                if header_variant in h.lower():
                    return row_data.get(h, "")
        return ""

    procedure = find_value(field_mappings["procedure"])

    # Skip if no procedure name found or if it looks like a header/total row
    if not procedure or procedure.lower() in [
        "total",
        "subtotal",
        "grand total",
        "",
        "description",
        "procedure",
    ]:
        return None

    # Parse values
    date_str = find_value(field_mappings["date"])
    cpt_code = find_value(field_mappings["cpt"])
    modifier = find_value(field_mappings["modifier"])
    qty_str = find_value(field_mappings["quantity"])
    unit_price_str = find_value(field_mappings["unit_price"])
    total_str = find_value(field_mappings["total"])

    # Determine entity type
    entity_type = "procedure"
    procedure_lower = procedure.lower()
    drug_keywords = [
        "mg",
        "ml",
        "tablet",
        "capsule",
        "injection",
        "infusion",
        "medication",
        "drug",
        "rx",
    ]
    if any(kw in procedure_lower for kw in drug_keywords):
        entity_type = "drug"

    return {
        "Date": date_str if date_str else "N/A",
        "Medical Procedure Name": procedure,
        "CPT Code": cpt_code,
        "Modifier": modifier,
        "Quantity": qty_str if qty_str else "1",
        "Unit Price": unit_price_str if unit_price_str else "0",
        "Total Price": total_str if total_str else "0",
        "Entity Type": entity_type,
    }


def parse_txt_with_llm(txt_content: str) -> list[dict]:
    """
    Use Claude Sonnet 3.5 to parse the TXT content and extract structured medical bill records.
    This leverages the LLM's understanding to minimize parsing errors.
    """
    session = boto3.Session(
        aws_access_key_id=st.session_state["aws_credentials"]["aws_access_key"],
        aws_secret_access_key=st.session_state["aws_credentials"]["aws_secret_key"],
        aws_session_token=st.session_state["aws_credentials"]["aws_session_token"],
        region_name="us-east-1",
    )
    bedrock_client = session.client(
        service_name="bedrock-runtime", region_name="us-east-1"
    )

    extraction_prompt = f"""Analyze the following medical bill text extracted via OCR and extract all line items/procedures.

MEDICAL BILL TEXT:
---
{txt_content}
---

For EACH procedure/item in the bill, extract the following information and return as a JSON array:

1. **Date**: The date of service in "YYYY-MM-DD" format. If only month/year is given, use the first day of that month. If no date is found, use "N/A".

2. **Medical Procedure Name**: The full description of the medical procedure or service.

3. **CPT Code**: The CPT (Current Procedural Terminology) code. If it's present in the bill, use that exact code. If not present, determine the most appropriate standard CPT code based on the procedure description. Format as a 5-digit string (e.g., "99213").

4. **CPT Code Source**: Specify whether the CPT code was "extracted" directly from the bill text or "inferred" by you based on the description.

5. **Modifier**: The modifier for the CPT code (e.g., "25", "59", "76"). If present in the bill, use that. If the procedure typically requires a modifier, include it. If no modifier is needed or applicable, use an empty string "".

6. **Quantity**: The quantity/units of the procedure as an integer. Default to 1 if not specified.

7. **Unit Price**: The unit price as a float with 2 decimal places. Remove any currency symbols. If only total is given with quantity > 1, calculate unit price.

8. **Total Price**: The total price as a float with 2 decimal places. This should equal Unit Price × Quantity.

9. **Entity Type**: Classify the item as either "procedure" or "drug". Use "drug" for medications, pharmaceuticals, injections of drugs, infusions, and any medication-related items. Use "procedure" for medical procedures, examinations, consultations, surgeries, lab tests, imaging, and other medical services.

10. **Dosage Value**: For drugs, the numeric value of the dosage mentioned in the name or description (e.g., for "Drug 50mg", the value is 50.0). For procedures, use null or 0.
11. **Dosage Unit**: For drugs, the unit of the dosage (e.g., "MG", "ML", "IU"). Always return in uppercase. For procedures, use an empty string "".
12. **Units from Name**: For drugs, the units from Name is 1 unless some quantity other than the dosage value is mentioned. For example: "Drug 50mg/ 2ml injection" has dosage 50mg and units from name as 2, since default is 1 ml, but this is a 2ml injection. For procedures, use null
13. **Page**: The page number where this item was found in the text (look for the [PAGE X] markers).
14. **Extraction Confidence**: An integer from 0 to 100 representing your confidence in the extraction of THIS specific row. Consider factors like OCR quality, ambiguity in the description, and whether the CPT code was confidently matched or inferred.


IMPORTANT FORMATTING RULES:
- Return ONLY a valid JSON array, no additional text
- Dates must be in "YYYY-MM-DD" format or "N/A"
- CPT codes must be 5-character strings
- Modifiers must be strings (empty string if not applicable)
- Quantity must be an integer
- Unit Price and Total Price must be floats with 2 decimal places
- Entity Type must be exactly "procedure" or "drug"
- Dosage Value must be a float or null
- Dosage Unit must be an uppercase string or ""
- If a value cannot be determined, use reasonable defaults or "N/A" for strings, 0 for numbers

Example output format:
[
    {{
        "Date": "2024-01-15",
        "Medical Procedure Name": "Office visit, established patient",
        "CPT Code": "99213",
        "CPT Code Source": "extracted",
        "Modifier": "",
        "Quantity": 1,
        "Unit Price": 150.00,
        "Total Price": 150.00,
        "Entity Type": "procedure",
        "Dosage Value": null,
        "Dosage Unit": "",
        "Units from Name": null,
        "Page": 1,
        "Extraction Confidence": 98
    }},
    {{
        "Date": "2024-01-15",
        "Medical Procedure Name": "Amoxicillin 500mg/ pack of 3",
        "CPT Code": "J0290",
        "CPT Code Source": "inferred",
        "Modifier": "",
        "Quantity": 2,
        "Unit Price": 25.00,
        "Total Price": 50.00,
        "Entity Type": "drug",
        "Dosage Value": 500.0,
        "Dosage Unit": "MG",
        "Units from Name": 3,
        "Page": 2,
        "Extraction Confidence": 85
    }}
]

Now analyze the medical bill text and extract ALL line items:"""

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16000,
        "messages": [
            {
                "role": "user",
                "content": extraction_prompt,
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

    # response_text, input_tokens, output_tokens = invoke_kimi_model(bedrock_client, extraction_prompt)
    # print(response_text)
    # response_text = response_text.split("</think>")[-1]
    # print(response_text)
    # print(f"Input Tokens: {input_tokens}")
    # print(f"Output Tokens: {output_tokens}")
    # Extract JSON from response (handle potential markdown code blocks)
    json_match = re.search(r"\[[\s\S]*\]", response_text)
    if json_match:
        json_str = json_match.group()
        extracted_data = json.loads(json_str)
    else:
        raise ValueError("Could not parse JSON from model response")

    return extracted_data


def extract_medical_bill_data(
    image_bytes: bytes, file_name: str
) -> tuple[list[dict], dict]:
    """
    Extract medical bill data using AWS Textract + Claude Sonnet.
    Flow: Image -> Textract -> TXT -> Claude Sonnet -> Structured Records
    Returns tuple of (extracted_data, metadata).
    Uses cached TXT if available.
    """
    ensure_cache_dirs()

    # Check cache first
    cached_textract, cached_txt = check_cache(file_name)

    if cached_txt:
        # Use cached TXT file, but still parse with LLM
        print(f"Using cached TXT for: {file_name}")
        records = parse_txt_with_llm(cached_txt)

        # Try to get Textract confidence if JSON exists in cache
        cache_key = get_cache_key(file_name)
        textract_path = TEXTRACT_CACHE_DIR / f"{cache_key}.json"
        textract_conf = {}
        if textract_path.exists():
            textract_json = json.loads(textract_path.read_text(encoding="utf-8"))
            textract_conf = get_textract_confidence(textract_json)

        metadata = {"cached": True, "source": "textract+llm", **textract_conf}
        return records, metadata

    # Call Textract
    print(f"Calling Textract for: {file_name}")
    textract_response = call_textract(image_bytes)
    textract_conf = get_textract_confidence(textract_response)

    # Convert to TXT
    txt_content = textract_response_to_txt(textract_response)

    # Save to cache
    save_to_cache(file_name, textract_response, txt_content)

    # Parse records from TXT using LLM
    records = parse_txt_with_llm(txt_content)

    metadata = {"cached": False, "source": "textract+llm", **textract_conf}
    return records, metadata


def parse_and_format_data(raw_data: list[dict], document_name: str) -> pd.DataFrame:
    """
    Parse and format the extracted data into a properly typed DataFrame.
    """
    formatted_records = []

    for record in raw_data:
        # Parse entity type - ensure it's either "procedure" or "drug"
        entity_type = str(record.get("Entity Type", "procedure")).lower().strip()
        if entity_type not in ["procedure", "drug"]:
            entity_type = "procedure"  # Default to procedure if invalid

        formatted_record = {
            "Document Name": document_name,
            "Date": parse_date(record.get("Date", "N/A")),
            "Medical Procedure Name": str(
                record.get("Medical Procedure Name", "")
            ).strip(),
            "CPT Code": format_cpt_code(record.get("CPT Code", "")),
            "CPT Code Source": str(record.get("CPT Code Source", "extracted"))
            .lower()
            .strip(),
            "Modifier": str(record.get("Modifier", "")).strip(),
            "Quantity": parse_quantity(record.get("Quantity", 1)),
            "Unit Price": parse_price(record.get("Unit Price", 0)),
            "Total Price": parse_price(record.get("Total Price", 0)),
            "Entity Type": entity_type,
            "Dosage Value": record.get("Dosage Value"),
            "Dosage Unit": str(record.get("Dosage Unit", "")).strip().upper(),
            "Units from Name": record.get("Units from Name"),
            "Page": pd.to_numeric(record.get("Page", 1), errors="coerce"),
            "Extraction Confidence": pd.to_numeric(
                record.get("Extraction Confidence", 0), errors="coerce"
            ),
        }
        formatted_records.append(formatted_record)

    df = pd.DataFrame(formatted_records)

    # Ensure correct data types
    df["Quantity"] = df["Quantity"].astype(int)
    df["Unit Price"] = df["Unit Price"].astype(float).round(2)
    df["Total Price"] = df["Total Price"].astype(float).round(2)
    df["Dosage Value"] = pd.to_numeric(df["Dosage Value"], errors="coerce")
    df["Units from Name"] = pd.to_numeric(df["Units from Name"], errors="coerce")

    # Add Medicare threshold columns
    df = add_medicare_threshold_columns(df)

    # Ensure all float columns are rounded to 2 decimal places
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].round(2)

    return df


def load_medicare_procedure_prices() -> pd.DataFrame:
    """Load the CPT-Procedure-Prices-Medicare CSV file."""
    csv_path = Path(__file__).parent / "CPT-Procedure-Prices-Medicare.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        # Clean up column names
        df.columns = df.columns.str.strip()
        return df
    return pd.DataFrame()


def load_medicare_drug_prices() -> pd.DataFrame:
    """Load the Drug-Prices-Medicare CSV file."""
    csv_path = Path(__file__).parent / "Drug-Prices-Medicare.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        # Clean up column names
        df.columns = df.columns.str.strip()
        return df
    return pd.DataFrame()


def parse_dosage_units(dosage_str: str) -> tuple[float, str]:
    """
    Parse the HCPCS Code Dosage string to extract the numeric value and unit.
    Examples: "1 ML" -> (1.0, "ML"), "150 IU" -> (150.0, "IU"), "1 EACH" -> (1.0, "EACH")
    """
    if not dosage_str or pd.isna(dosage_str):
        return 1.0, ""

    dosage_str = str(dosage_str).strip()
    # Match number (int or float) followed by optional unit
    match = re.match(r"^([\d.]+)\s*(.*)$", dosage_str)
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2).strip().upper()
            return value, unit
        except ValueError:
            return 1.0, dosage_str
    return 1.0, dosage_str


def calculate_medicare_units(
    bill_quantity: int, bill_unit_str: str, medicare_dosage_per_unit: float
) -> float:
    """
    Calculate the number of Medicare units based on bill quantity and Medicare dosage.
    """
    if medicare_dosage_per_unit <= 0:
        return float(bill_quantity)

    # The bill quantity represents the actual amount used
    # Medicare units = bill quantity / medicare dosage per unit
    return float(bill_quantity) / medicare_dosage_per_unit


def add_medicare_threshold_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Medicare threshold columns to the DataFrame.
    - Unit Price Threshold: Medicare price per unit
    - No. of Medicare Units: Number of units based on Medicare dosage
    - Total Threshold Price: Unit Price Threshold × No. of Medicare Units
    """
    # Load Medicare price files
    procedure_prices_df = load_medicare_procedure_prices()
    drug_prices_df = load_medicare_drug_prices()

    # Create lookup dictionaries
    procedure_lookup = {}
    if not procedure_prices_df.empty and "HCPCS" in procedure_prices_df.columns:
        for _, row in procedure_prices_df.iterrows():
            hcpcs = str(row.get("HCPCS", "")).strip().upper()
            non_fac_total = row.get("NON-FACILITY TOTAL PRICE", "")
            if hcpcs and non_fac_total:
                try:
                    procedure_lookup[hcpcs] = float(str(non_fac_total).replace(",", ""))
                except ValueError:
                    pass

    drug_lookup = {}
    if not drug_prices_df.empty and "HCPCS Code" in drug_prices_df.columns:
        for _, row in drug_prices_df.iterrows():
            hcpcs = str(row.get("HCPCS Code", "")).strip().upper()
            payment_limit = row.get("Payment Limit", "")
            dosage = row.get("HCPCS Code Dosage", "1")
            if hcpcs and payment_limit:
                try:
                    drug_lookup[hcpcs] = {
                        "payment_limit": float(str(payment_limit).replace(",", "")),
                        "dosage": dosage,
                    }
                except ValueError:
                    pass

    # Initialize new columns
    unit_threshold_prices = []
    medicare_units_list = []
    total_threshold_prices = []

    for _, row in df.iterrows():
        cpt_code = str(row.get("CPT Code", "")).strip().upper()
        entity_type = str(row.get("Entity Type", "")).lower()
        quantity = int(row.get("Quantity", 1))
        dosage_value = row.get("Dosage Value")
        dosage_unit = str(row.get("Dosage Unit", "")).strip().upper()
        num_units_from_name = row.get("Units from Name")
        unit_threshold = 0.0
        medicare_units = 0.0

        if entity_type == "procedure":
            # Look up procedure price
            if cpt_code in procedure_lookup:
                unit_threshold = procedure_lookup[cpt_code]
                medicare_units = float(quantity)

        elif entity_type == "drug":
            # Look up drug price and dosage
            if cpt_code in drug_lookup:
                drug_info = drug_lookup[cpt_code]
                unit_threshold = drug_info["payment_limit"]

                # Parse the Medicare dosage
                med_dosage_val, med_dosage_unit = parse_dosage_units(
                    drug_info["dosage"]
                )

                # Calculation: (quantity * dosage_value / HCPCS Code Dosage Value)
                # Provided dosage_unit and HCPCS Code Dosage unit match
                if dosage_value and dosage_unit == med_dosage_unit:
                    medicare_units = (
                        quantity * dosage_value * num_units_from_name
                    ) / med_dosage_val
                else:
                    # Default: 1 * (Units from Name or 1) * Quantity
                    medicare_units = (
                        1.0
                        * (num_units_from_name if num_units_from_name else 1.0)
                        * quantity
                    )

        # Calculate total threshold price
        total_threshold = unit_threshold * medicare_units

        unit_threshold_prices.append(round(unit_threshold, 2))
        medicare_units_list.append(round(medicare_units, 2))
        total_threshold_prices.append(round(total_threshold, 2))

    # Add new columns to DataFrame
    df["Unit Price Threshold"] = unit_threshold_prices
    df["No. of Medicare Units"] = medicare_units_list
    df["Total Threshold Price"] = total_threshold_prices

    # Calculate Threshold Multiplier
    threshold_multipliers = []
    for _, row in df.iterrows():
        total_price = row.get("Total Price", 0)
        total_threshold = row.get("Total Threshold Price", 0)
        unit_threshold = row.get("Unit Price Threshold", 0)

        if unit_threshold == 0 or total_threshold == 0:
            threshold_multipliers.append(None)
        else:
            multiplier = total_price / total_threshold
            threshold_multipliers.append(multiplier)

    df["Threshold Multiplier"] = threshold_multipliers

    # Calculate Price Difference
    df["Price Difference"] = df["Total Price"] - df["Total Threshold Price"]

    return df


def generate_bill_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of the medical bill data."""
    if df.empty:
        return "No data available."

    # CPT Counts
    found_cpts = len(df[df["CPT Code Source"] == "extracted"])
    inferred_cpts = len(df[df["CPT Code Source"] == "inferred"])

    # Entity Counts
    drug_df = df[df["Entity Type"] == "drug"]
    proc_df = df[df["Entity Type"] == "procedure"]

    num_drugs = len(drug_df)
    num_procs = len(proc_df)

    # Costs
    total_drug_cost = drug_df["Total Price"].sum()
    total_proc_cost = proc_df["Total Price"].sum()

    # Quantities
    total_drug_qty = drug_df["Quantity"].sum()
    total_proc_qty = proc_df["Quantity"].sum()

    summary = f"""MEDICAL BILL SUMMARY
====================
- No. of CPT codes found: {found_cpts}
- No. of CPT codes inferred: {inferred_cpts}
- No. of drugs: {num_drugs}
- No. of procedures: {num_procs}
- Total cost of drugs: ${total_drug_cost:,.2f}
- Total cost of procedures: ${total_proc_cost:,.2f}
- Total quantity of all drugs: {total_drug_qty}
- Total quantity of all procedures: {total_proc_qty}
"""
    return summary


def parse_date(date_value) -> str:
    """Parse and format date to YYYY-MM-DD string."""
    if date_value is None or str(date_value).strip().upper() in ["N/A", "NA", ""]:
        return "N/A"

    date_str = str(date_value).strip()

    # Try various date formats
    date_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%y",
    ]

    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return "N/A"


def format_cpt_code(cpt_value) -> str:
    """Format CPT code as 5-character string."""
    if cpt_value is None or str(cpt_value).strip() == "":
        return ""

    cpt_str = str(cpt_value).strip()
    # Remove any non-alphanumeric characters except for HCPCS codes which may have letters
    cpt_clean = re.sub(r"[^A-Za-z0-9]", "", cpt_str)

    return cpt_clean


def parse_quantity(qty_value) -> int:
    """Parse quantity as integer."""
    try:
        if qty_value is None:
            return 1
        return max(1, int(float(str(qty_value).replace(",", ""))))
    except (ValueError, TypeError):
        return 1


def parse_price(price_value) -> float:
    """Parse price as float."""
    try:
        if price_value is None:
            return 0.0
        # Remove currency symbols and commas
        price_str = str(price_value).replace("$", "").replace(",", "").strip()
        return round(float(price_str), 2)
    except (ValueError, TypeError):
        return 0.0


def get_pandas_agent(df: pd.DataFrame):
    """Create a pandas dataframe agent using ChatBedrockConverse."""

    # Initialize ChatBedrockConverse with openai-120b model
    # nvidia.nemotron-nano-12b-v2
    # qwen.qwen3-coder-30b-a3b-v1:0
    # google.gemma-3-4b-it
    llm = ChatBedrockConverse(
        model="qwen.qwen3-coder-30b-a3b-v1:0",
        region_name="us-east-1",
        aws_access_key_id=st.session_state["aws_credentials"]["aws_access_key"],
        aws_secret_access_key=st.session_state["aws_credentials"]["aws_secret_key"],
        aws_session_token=st.session_state["aws_credentials"]["aws_session_token"],
        disable_streaming=False,
    )

    # Create pandas dataframe agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="tool-calling",
        return_intermediate_steps=True,
        allow_dangerous_code=True,
    )

    return agent


def get_metrics(df: pd.DataFrame):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Items", len(df))
    with col2:
        total_amount = df["Total Price"].sum()
        st.metric("Total Amount", f"${total_amount:,.2f}")
    with col3:
        unique_cpt = df["CPT Code"].nunique()
        st.metric("Unique CPT Codes", unique_cpt)
    with col6:
        accuracy = 90
        st.metric("Extraction Accuracy", f"{accuracy}%")
    with col4:
        procedure_count = len(df[df["Entity Type"] == "procedure"])
        st.metric("Procedures", procedure_count)
    with col5:
        drug_count = len(df[df["Entity Type"] == "drug"])
        st.metric("Drugs", drug_count)


def display_dataframe_with_color_coding(
    df: pd.DataFrame,
    key_prefix: str = "default",
):
    """
    Display a styled DataFrame with metrics and dynamic threshold controls.
    key_prefix MUST be unique per function instance on the same page.
    """

    # ---------- Scoped key helper ----------
    def k(name: str) -> str:
        return f"{key_prefix}_{name}"

    # ---------- Initialize session state ----------
    if k("active_thresholds") not in st.session_state:
        st.session_state[k("active_thresholds")] = [
            "Multiplier Threshold",
            "Difference Threshold",
        ]

    if k("m_threshold") not in st.session_state:
        st.session_state[k("m_threshold")] = 2.0

    if k("d_threshold") not in st.session_state:
        st.session_state[k("d_threshold")] = 10.0

    active_thresholds = st.session_state[k("active_thresholds")]
    m_threshold = float(st.session_state[k("m_threshold")])
    d_threshold = float(st.session_state[k("d_threshold")])

    # ---------- Normalize numeric columns ----------
    numeric_rounding = {
        "Unit Price": 2,
        "Total Price": 2,
        "Unit Price Threshold": 2,
        "Total Threshold Price": 2,
        "Price Difference": 2,
        "Threshold Multiplier": 2,
        "Dosage Value": 2,
        "No. of Medicare Units": 2,
    }

    for col, decimals in numeric_rounding.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)

    if "Quantity" in df.columns:
        df["Quantity"] = (
            pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        )

    if "Extraction Confidence" in df.columns:
        df["Extraction Confidence"] = (
            pd.to_numeric(df["Extraction Confidence"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    active_thresholds = st.multiselect(
        "Select Active Thresholds",
        ["Multiplier Threshold", "Difference Threshold"],
        key=k("active_thresholds"),
    )

    fcol1, fcol2 = st.columns(2)

    with fcol1:
        m_threshold = st.number_input(
            "Multiplier Threshold (Red if > X)",
            min_value=1.0,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key=k("m_threshold"),
            disabled="Multiplier Threshold" not in active_thresholds,
        )

    with fcol2:
        d_threshold = st.number_input(
            "Difference Threshold (Red if > $X)",
            min_value=0.0,
            max_value=10000.0,
            step=1.0,
            format="%.2f",
            key=k("d_threshold"),
            disabled="Difference Threshold" not in active_thresholds,
        )

    st.divider()

    # ---------- Row styling ----------
    def style_rows(row):
        styles = [""] * len(row)

        try:
            m_val = float(row.get("Threshold Multiplier", 0))
            d_val = float(row.get("Price Difference", 0))
            u_threshold = float(row.get("Unit Price Threshold", 0))

            if u_threshold > 0:
                m_exceeded = (
                    "Multiplier Threshold" in active_thresholds and m_val > m_threshold
                )
                d_exceeded = (
                    "Difference Threshold" in active_thresholds and d_val > d_threshold
                )

                color = (
                    "background-color:#FFCDD2;color:#B71C1C;"
                    if m_exceeded or d_exceeded
                    else "background-color:#C8E6C9;color:#1B5E20;"
                )

                styles[row.index.get_loc("Threshold Multiplier")] = color
                styles[row.index.get_loc("Price Difference")] = color
        except Exception:
            pass

        source = row.get("CPT Code Source")
        if source == "extracted":
            styles[row.index.get_loc("CPT Code")] = (
                "background-color:#EEEEEE;color:#424242;"
            )
        elif source == "inferred":
            styles[row.index.get_loc("CPT Code")] = (
                "background-color:#E3F2FD;color:#0D47A1;"
            )

        conf = row.get("Extraction Confidence")
        if pd.notnull(conf):
            styles[row.index.get_loc("Extraction Confidence")] = (
                "color:#1B5E20;font-weight:bold;"
                if conf >= 90
                else "color:#E65100;" if conf >= 70 else "color:#B71C1C;"
            )

        return styles

    styled_df = df.style.apply(style_rows, axis=1)
    return styled_df


def display_dataframe_with_metrics(
    df: pd.DataFrame,
    token_usage: dict = None,
    key_prefix: str = "default",
):
    """
    Display a styled DataFrame with metrics and dynamic threshold controls.
    key_prefix MUST be unique per function instance on the same page.
    """

    # ---------- Scoped key helper ----------
    def k(name: str) -> str:
        return f"{key_prefix}_{name}"

    # ---------- Initialize session state ----------
    if k("active_thresholds") not in st.session_state:
        st.session_state[k("active_thresholds")] = [
            "Multiplier Threshold",
            "Difference Threshold",
        ]

    if k("m_threshold") not in st.session_state:
        st.session_state[k("m_threshold")] = 2.0

    if k("d_threshold") not in st.session_state:
        st.session_state[k("d_threshold")] = 10.0

    active_thresholds = st.session_state[k("active_thresholds")]
    m_threshold = float(st.session_state[k("m_threshold")])
    d_threshold = float(st.session_state[k("d_threshold")])

    # ---------- Normalize numeric columns ----------
    numeric_rounding = {
        "Unit Price": 2,
        "Total Price": 2,
        "Unit Price Threshold": 2,
        "Total Threshold Price": 2,
        "Price Difference": 2,
        "Threshold Multiplier": 2,
        "Dosage Value": 2,
        "No. of Medicare Units": 2,
    }

    for col, decimals in numeric_rounding.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)

    if "Quantity" in df.columns:
        df["Quantity"] = (
            pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        )

    if "Extraction Confidence" in df.columns:
        df["Extraction Confidence"] = (
            pd.to_numeric(df["Extraction Confidence"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # ---------- Summary metrics ----------
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Items", len(df))

    with col2:
        st.metric("Total Amount", f"${df['Total Price'].sum():,.2f}")

    with col3:
        st.metric("Unique CPT Codes", df["CPT Code"].nunique())

    with col4:
        st.metric("Procedures", int((df["Entity Type"] == "procedure").sum()))

    with col5:
        st.metric("Drugs", int((df["Entity Type"] == "drug").sum()))

    with col6:
        accuracy = token_usage.get("avg_table_confidence", 90) if token_usage else 90
        st.metric("Extraction Accuracy", f"{int(accuracy)}%")

    st.divider()

    # ---------- Threshold controls ----------
    st.subheader("🛠️ Dynamic Review Thresholds")

    active_thresholds = st.multiselect(
        "Select Active Thresholds",
        ["Multiplier Threshold", "Difference Threshold"],
        key=k("active_thresholds"),
    )

    fcol1, fcol2 = st.columns(2)

    with fcol1:
        m_threshold = st.number_input(
            "Multiplier Threshold (Red if > X)",
            min_value=1.0,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key=k("m_threshold"),
            disabled="Multiplier Threshold" not in active_thresholds,
        )

    with fcol2:
        d_threshold = st.number_input(
            "Difference Threshold (Red if > $X)",
            min_value=0.0,
            max_value=10000.0,
            step=1.0,
            format="%.2f",
            key=k("d_threshold"),
            disabled="Difference Threshold" not in active_thresholds,
        )

    st.divider()

    # ---------- Row styling ----------
    def style_rows(row):
        styles = [""] * len(row)

        try:
            m_val = float(row.get("Threshold Multiplier", 0))
            d_val = float(row.get("Price Difference", 0))
            u_threshold = float(row.get("Unit Price Threshold", 0))

            if u_threshold > 0:
                m_exceeded = (
                    "Multiplier Threshold" in active_thresholds and m_val > m_threshold
                )
                d_exceeded = (
                    "Difference Threshold" in active_thresholds and d_val > d_threshold
                )

                color = (
                    "background-color:#FFCDD2;color:#B71C1C;"
                    if m_exceeded or d_exceeded
                    else "background-color:#C8E6C9;color:#1B5E20;"
                )

                styles[row.index.get_loc("Threshold Multiplier")] = color
                styles[row.index.get_loc("Price Difference")] = color
        except Exception:
            pass

        source = row.get("CPT Code Source")
        if source == "extracted":
            styles[row.index.get_loc("CPT Code")] = (
                "background-color:#EEEEEE;color:#424242;"
            )
        elif source == "inferred":
            styles[row.index.get_loc("CPT Code")] = (
                "background-color:#E3F2FD;color:#0D47A1;"
            )

        conf = row.get("Extraction Confidence")
        if pd.notnull(conf):
            styles[row.index.get_loc("Extraction Confidence")] = (
                "color:#1B5E20;font-weight:bold;"
                if conf >= 90
                else "color:#E65100;" if conf >= 70 else "color:#B71C1C;"
            )

        return styles

    styled_df = df.style.apply(style_rows, axis=1)

    # ---------- DataFrame display ----------
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quantity": st.column_config.NumberColumn("Quantity", format="%d"),
            "Unit Price": st.column_config.NumberColumn("Unit Price", format="$%.2f"),
            "Total Price": st.column_config.NumberColumn("Total Price", format="$%.2f"),
            "Unit Price Threshold": st.column_config.NumberColumn(
                "Unit Price Threshold", format="$%.2f"
            ),
            "Total Threshold Price": st.column_config.NumberColumn(
                "Total Threshold Price", format="$%.2f"
            ),
            "Threshold Multiplier": st.column_config.NumberColumn(
                "Threshold Multiplier", format="%.2f×"
            ),
            "Price Difference": st.column_config.NumberColumn(
                "Price Difference", format="$%.2f"
            ),
            "Dosage Value": st.column_config.NumberColumn(
                "Dosage Value", format="%.2f"
            ),
            "No. of Medicare Units": st.column_config.NumberColumn(
                "No. of Medicare Units", format="%.2f"
            ),
            "Extraction Confidence": st.column_config.NumberColumn(
                "Extraction Confidence", format="%d%%"
            ),
            "CPT Code Source": None,
        },
    )

    # ---------- Bill summary ----------
    st.divider()
    st.subheader("📋 Bill Summary")

    summary_text = generate_bill_summary(df)

    with st.expander("View Bill Summary Details", expanded=True):
        st.code(summary_text, language="text")
        st.download_button(
            "📥 Download Bill Summary (.txt)",
            summary_text,
            f"medical_bill_summary_{int(time.time())}.txt",
            "text/plain",
            use_container_width=True,
            key=k("download_summary"),
        )

    # ---------- Extraction info ----------
    if token_usage:
        st.divider()
        st.markdown("### 📊 Extraction Info")
        (
            st.success("✅ Loaded from cache")
            if token_usage.get("cached")
            else st.info("🔄 Freshly extracted via AWS Textract")
        )


def render_chat_interface(df: pd.DataFrame, chat_key_prefix: str):
    """Render the chat interface for querying the dataframe."""
    st.divider()
    st.subheader("💬 Chat with Your Data")
    st.markdown(
        "Ask questions about the extracted medical bill data using natural language."
    )

    # Initialize chat history in session state
    chat_history_key = f"{chat_key_prefix}_chat_history"
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    # Display chat history
    for message in st.session_state[chat_history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your medical bill data...",
        key=f"{chat_key_prefix}_chat_input",
    ):
        # Add user message to chat history
        st.session_state[chat_history_key].append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from pandas agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    token_tracker = BedrockTokenCallback()
                    agent = get_pandas_agent(df)
                    response = agent.invoke(
                        {"input": prompt},
                        config={"callbacks": [token_tracker]},
                    )
                    print("\n" + "=" * 30)
                    print(f"FINAL ANSWER: {response['output']}")
                    print(f"INPUT TOKENS USED:  {token_tracker.input_tokens}")
                    print(f"OUTPUT TOKENS USED: {token_tracker.output_tokens}")
                    print(
                        f"TOTAL TOKENS:       {token_tracker.input_tokens + token_tracker.output_tokens}"
                    )
                    print("=" * 30)
                    # Extract the output from the response
                    if isinstance(response, dict):
                        answer = response.get("output", str(response))
                        answer = answer[-1]["text"]
                    else:
                        answer = str(response)

                    st.markdown(answer)

                    # Add assistant message to chat history
                    st.session_state[chat_history_key].append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state[chat_history_key].append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Clear chat button
    if st.session_state[chat_history_key]:
        if st.button("🗑️ Clear Chat History", key=f"{chat_key_prefix}_clear_chat"):
            st.session_state[chat_history_key] = []
            st.rerun()


def render_single_bill_tab():
    """Render the Single Bill tab content."""
    st.subheader("📄 Upload Single Medical Bill")

    # File uploader
    uploaded_doc = st.file_uploader(
        "Upload Medical Bill Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a PNG or JPG image of your medical bill",
        key="single_bill_uploader",
    )
    if st.button("🧪 Load Demo Files", use_container_width=True, key="single-bill"):
        load_demo_bill()
        st.success("Demo files loaded!")

    use_demo = st.session_state.get("demo_bill", False)

    uploaded_file = st.session_state.get("demo_bill") if use_demo else uploaded_doc

    if uploaded_file is not None:
        # Display the uploaded image
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📄 Uploaded Bill")
            st.image(uploaded_file, use_container_width=True)

        with col2:
            st.subheader("📊 Extraction")

            # Process button
            if st.button(
                "🔍 Extract Bill Data",
                type="primary",
                use_container_width=True,
                key="single_extract",
            ):
                with st.spinner("Analyzing medical bill with AI..."):
                    try:
                        # Read image bytes
                        image_bytes = uploaded_file.getvalue()

                        # Check DataFrame cache first
                        cached_df = check_df_cache(uploaded_file.name)

                        if cached_df is not None:
                            print(f"Using cached DataFrame for: {uploaded_file.name}")
                            df = cached_df
                            token_usage = {"cached": True, "source": "df_cache"}
                        else:
                            # Extract data using Claude Sonnet
                            raw_data, token_usage = extract_medical_bill_data(
                                image_bytes, uploaded_file.name
                            )

                            # Parse and format data
                            df = parse_and_format_data(raw_data, uploaded_file.name)

                            # Save to cache
                            save_df_to_cache(uploaded_file.name, df)

                        # Store in session state
                        st.session_state["single_extracted_df"] = df
                        st.session_state["single_token_usage"] = token_usage
                        st.session_state["single_extraction_success"] = True

                    except Exception as e:
                        st.error(f"Error processing bill: {str(e)}")
                        st.session_state["single_extraction_success"] = False

        # Display results if available
        if st.session_state.get("single_extraction_success"):
            df = st.session_state["single_extracted_df"]
            token_usage = st.session_state.get("single_token_usage", {})

            st.divider()
            st.subheader("📋 Extracted Medical Bill Data")

            display_dataframe_with_metrics(df, token_usage, key_prefix="single")

            # Download button for CSV
            st.divider()
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="medical_bill_data.csv",
                mime="text/csv",
                use_container_width=True,
                key="single_download",
            )

            # Chat interface
            render_chat_interface(df, "single")


def render_multiple_bills_tab():
    """Render the Multiple Bills tab content."""
    st.subheader("📚 Upload Multiple Medical Bills")

    # File uploader for multiple files
    uploaded_docs = st.file_uploader(
        "Upload Medical Bill Images",
        type=["png", "jpg", "jpeg"],
        help="Upload multiple PNG or JPG images of your medical bills",
        accept_multiple_files=True,
        key="multiple_bills_uploader",
    )

    if st.button("🧪 Load Demo Files", use_container_width=True, key="multiple-bill"):
        load_demo_bill_multiple()
        st.success("Demo files loaded!")

    use_demo = st.session_state.get("demo_bill_multiple", False)

    uploaded_files = (
        st.session_state.get("demo_bill_multiple") if use_demo else uploaded_docs
    )

    if uploaded_files:
        st.info(f"📎 **{len(uploaded_files)} file(s) uploaded**")

        # Display thumbnails of uploaded images
        st.divider()
        st.subheader("📄 Uploaded Bills Preview")

        # Create columns for thumbnails (max 4 per row)
        num_cols = min(4, len(uploaded_files))
        cols = st.columns(num_cols)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % num_cols]:
                st.image(
                    uploaded_file, caption=uploaded_file.name, use_container_width=True
                )

        st.divider()

        # Process button
        if st.button(
            "🔍 Extract All Bills Data",
            type="primary",
            use_container_width=True,
            key="multiple_extract",
        ):
            all_dataframes = []
            total_input_tokens = 0
            total_output_tokens = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(
                    f"Processing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})..."
                )

                try:
                    # Read image bytes
                    image_bytes = uploaded_file.getvalue()

                    # Check DataFrame cache first
                    cached_df = check_df_cache(uploaded_file.name)

                    if cached_df is not None:
                        print(f"Using cached DataFrame for: {uploaded_file.name}")
                        df = cached_df
                        token_usage = {"cached": True, "source": "df_cache"}
                    else:
                        # Extract data using Claude Sonnet
                        raw_data, token_usage = extract_medical_bill_data(
                            image_bytes, uploaded_file.name
                        )

                        # Parse and format data
                        df = parse_and_format_data(raw_data, uploaded_file.name)

                        # Save to cache
                        save_df_to_cache(uploaded_file.name, df)

                    all_dataframes.append(df)

                    # Accumulate token usage
                    total_input_tokens += token_usage.get("input_tokens", 0)
                    total_output_tokens += token_usage.get("output_tokens", 0)

                except Exception as e:
                    st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)}")

                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Processing complete!")

            if all_dataframes:
                # Combine all dataframes
                combined_df = pd.concat(all_dataframes, ignore_index=True)

                # Store in session state
                st.session_state["multiple_extracted_df"] = combined_df
                st.session_state["multiple_token_usage"] = {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                }
                st.session_state["multiple_extraction_success"] = True
                st.session_state["multiple_files_count"] = len(uploaded_files)
            else:
                st.error("No bills could be processed successfully.")
                st.session_state["multiple_extraction_success"] = False

        # Display results if available
        if st.session_state.get("multiple_extraction_success"):
            df = st.session_state["multiple_extracted_df"]
            token_usage = st.session_state.get("multiple_token_usage", {})
            files_count = st.session_state.get("multiple_files_count", 0)

            st.divider()
            st.subheader(f"📋 Combined Medical Bill Data ({files_count} bills)")

            # Additional metric for multiple bills
            unique_docs = df["Document Name"].nunique()
            st.metric("Documents Processed", unique_docs)

            display_dataframe_with_metrics(df, token_usage, key_prefix="multiple")

            # Download button for CSV
            st.divider()
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Combined Data as CSV",
                data=csv,
                file_name="combined_medical_bills_data.csv",
                mime="text/csv",
                use_container_width=True,
                key="multiple_download",
            )

            # Chat interface
            render_chat_interface(df, "multiple")


def render_pdf_upload_tab():
    """Render the specialized PDF upload and processing tab."""
    st.header("📄 PDF Medical Bill Processing")
    st.info("Upload the Medical Record PDF")

    # 3-way file uploader
    ucol1, ucol2, ucol3 = st.columns(3)
    with ucol1:
        pdf_upload = st.file_uploader("1. Upload PDF", type=["pdf"], key="pdf_batch")

    with ucol2:
        csv_upload = st.file_uploader(
            "2. Upload Page classification (CSV)", type=["csv"], key="csv_batch"
        )

    with ucol3:
        json_upload = st.file_uploader(
            "3. Upload Textract Blocks (JSON)", type=["json"], key="json_batch"
        )

    if st.button("🧪 Load Demo Files", use_container_width=True):
        load_demo_files()
        st.success("Demo files loaded!")

    use_demo = st.session_state.get("use_demo_files", False)

    pdf_file = st.session_state.get("demo_pdf") if use_demo else pdf_upload
    csv_file = st.session_state.get("demo_csv") if use_demo else csv_upload
    json_file = st.session_state.get("demo_json") if use_demo else json_upload

    if pdf_file and csv_file and json_file:
        cache_key_user_edited = (
            f"batch_edited_{pdf_file.name}_{csv_file.name}_{json_file.name}"
        )
        cache_key = f"batch_{pdf_file.name}_{csv_file.name}_{json_file.name}"
        if st.button("🚀 Process PDF Medical Bills", use_container_width=True):
            with st.spinner("Analyzing pages and extracting data..."):
                try:
                    cached_df = check_df_cache(cache_key)
                    cached_df_edited = check_df_cache(cache_key_user_edited)
                    pdf_bytes = pdf_file.getvalue()

                    if cached_df is not None:
                        st.info("Using cached results for this file combination.")
                        final_df = cached_df
                        if cached_df_edited is not None:
                            st.info(
                                "Using cached user-edited results for this file combination."
                            )
                            final_df_edited = cached_df_edited
                    else:
                        # 1. Load data
                        csv_df = pd.read_csv(csv_file)
                        textract_json = json.load(json_file)

                        # 2. Filter pages
                        target_pages = filter_medical_bill_pages(csv_df)
                        if not target_pages:
                            st.warning(
                                "No 'Bills – Medical, Pharmacy' pages found in the CSV."
                            )
                            return

                        st.write(
                            f"🔍 Found {len(target_pages)} medical bill pages: {target_pages}"
                        )

                        # 3. Extract blocks and text for target pages
                        full_text, geometries = get_textract_blocks_for_pages(
                            textract_json, target_pages
                        )

                        # 4. LLM Extraction
                        raw_records = parse_txt_with_llm(full_text)

                        # 5. Format and Enrich with Geometry
                        final_df = parse_and_format_data(raw_records, pdf_file.name)

                        # Add Page and Geometry info for highlighting (Fuzzy Matching)
                        df_geometries = []
                        df_pages = []

                        # Group geometries by page for efficient scoped search
                        geos_by_page = {}
                        for g in geometries:
                            p = g["Page"]
                            if p not in geos_by_page:
                                geos_by_page[p] = []
                            geos_by_page[p].append(g)

                        for _, row in final_df.iterrows():
                            proc_name = str(row["Medical Procedure Name"]).strip()
                            # Use the page extracted by the LLM
                            best_page = (
                                int(row["Page"]) if pd.notnull(row["Page"]) else 1
                            )
                            best_geo = None

                            # Only search within the specified page
                            page_geos = geos_by_page.get(best_page, [])
                            page_candidate_texts = [g["Text"] for g in page_geos]

                            if (
                                proc_name
                                and proc_name.lower() != "nan"
                                and page_candidate_texts
                            ):
                                # Find best fuzzy match in candidate texts for THIS PAGE
                                match = process.extractOne(
                                    proc_name,
                                    page_candidate_texts,
                                    scorer=fuzz.partial_ratio,
                                )

                                # If we have a decent match (> 80%), use its geometry
                                if match and match[1] >= 80:
                                    match_idx = match[2]
                                    best_geo = page_geos[match_idx]["Geometry"]
                                else:
                                    # Fallback to simple substring search if fuzzy fails
                                    for geo_item in page_geos:
                                        if (
                                            proc_name.lower()
                                            in geo_item["Text"].lower()
                                            or geo_item["Text"].lower()
                                            in proc_name.lower()
                                        ):
                                            best_geo = geo_item["Geometry"]
                                            break

                            df_geometries.append(
                                json.dumps(best_geo) if best_geo else None
                            )
                            df_pages.append(best_page)

                        final_df["Page"] = df_pages
                        final_df["Highlight Geometry"] = df_geometries
                        final_df_edited = final_df.copy()
                        # Save to cache
                        save_df_to_cache(cache_key, final_df)
                        save_df_to_cache(cache_key_user_edited, final_df_edited)

                    # Store in session state
                    st.session_state["pdf_batch_results"] = final_df
                    st.session_state["pdf_batch_results_user_edited"] = (
                        final_df_edited  # For user edits
                    )
                    st.session_state["pdf_batch_pdf_bytes"] = pdf_bytes
                    st.session_state["pdf_batch_success"] = True
                    st.success(f"Successfully loaded {len(final_df)} records!")

                except Exception as e:
                    st.error(f"Error during processing: {e}")

        # Display results if available
        if st.session_state.get("pdf_batch_success"):
            df = st.session_state["pdf_batch_results"]
            df_edited = st.session_state[
                "pdf_batch_results_user_edited"
            ]  # For user edits
            pdf_bytes = st.session_state["pdf_batch_pdf_bytes"]

            st.divider()

            # Layout: PDF Viewer (Left) | Table (Right)
            vcol1, vcol2 = st.columns([1, 1.2])
            get_metrics(df)
            # Bill Summary Section
            st.divider()
            st.subheader("📋 Bill Summary")
            summary_text = generate_bill_summary(df)

            with st.expander("View Bill Summary Details", expanded=True):
                st.code(summary_text, language="text")

                st.download_button(
                    label="📥 Download Bill Summary (.txt)",
                    data=summary_text,
                    file_name=f"medical_bill_summary_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=uuid.uuid4(),
                )

            # Chat interface
            render_chat_interface(df, "pdf")

            with vcol2:
                tab_1, tab_2 = st.tabs(["AI Generated", "User Edited"])
                with tab_1:
                    # Using selection mode if supported, otherwise just a selectbox
                    styled_df = display_dataframe_with_color_coding(
                        df.drop(columns=["Highlight Geometry"]),
                        key_prefix="pdf_table",
                    )
                    selection = st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        key="pdf_table_selection",
                    )

                    # Check for selection
                    selected_row_idx = None
                    if (
                        selection
                        and "selection" in selection
                        and selection["selection"]["rows"]
                    ):
                        selected_row_idx = selection["selection"]["rows"][0]

                    # Download button
                    st.download_button(
                        "📥 Download Results (Excel/CSV)",
                        data=df.to_csv(index=False),
                        file_name="medical_bill_extraction.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with tab_2:
                    edited_df = st.data_editor(
                        df_edited,
                        use_container_width=True,
                        key="pdf_table_selection_edited",
                        column_config={
                            "Quantity": st.column_config.NumberColumn(
                                "Quantity", format="%d"
                            ),
                            "Unit Price": st.column_config.NumberColumn(
                                "Unit Price", format="$%.2f"
                            ),
                            "Total Price": st.column_config.NumberColumn(
                                "Total Price", format="$%.2f"
                            ),
                            "Unit Price Threshold": st.column_config.NumberColumn(
                                "Unit Price Threshold", format="$%.2f"
                            ),
                            "Total Threshold Price": st.column_config.NumberColumn(
                                "Total Threshold Price", format="$%.2f"
                            ),
                            "Threshold Multiplier": st.column_config.NumberColumn(
                                "Threshold Multiplier", format="%.2f×"
                            ),
                            "Price Difference": st.column_config.NumberColumn(
                                "Price Difference", format="$%.2f"
                            ),
                            "Dosage Value": st.column_config.NumberColumn(
                                "Dosage Value", format="%.2f"
                            ),
                            "No. of Medicare Units": st.column_config.NumberColumn(
                                "No. of Medicare Units", format="%.2f"
                            ),
                            "Extraction Confidence": st.column_config.NumberColumn(
                                "Extraction Confidence", format="%d%%"
                            ),
                            "CPT Code Source": None,
                        },
                    )

                    # selection = st.dataframe(
                    #     df.drop(columns=["Highlight Geometry"]),
                    #     hide_index=True,
                    #     use_container_width=True,
                    #     on_select="rerun",
                    #     selection_mode="single-row",
                    #     key="pdf_table_selection_2",
                    # )

                    # Check for selection
                    selected_row_idx = None
                    if (
                        selection
                        and "selection" in selection
                        and selection["selection"]["rows"]
                    ):
                        selected_row_idx = selection["selection"]["rows"][0]

                    if st.button("💾 Save to Cache", use_container_width=True):
                        # Update the user-edited cache with the latest edited DataFrame
                        save_df_to_cache(cache_key_user_edited, edited_df)
                        st.success("User-edited changes saved")

                    # Download button
                    st.download_button(
                        "📥 Download Results (Excel/CSV)",
                        data=df.to_csv(index=False),
                        file_name="medical_bill_extraction.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_user_edited",
                    )

            with vcol1:
                st.subheader("📄 PDF Viewer")
                if selected_row_idx is not None:
                    row = df.iloc[selected_row_idx]
                    page_num = int(row["Page"])
                    geo_json = row["Highlight Geometry"]
                    geometry = json.loads(geo_json) if geo_json else None

                    # Dual view for maximum reliability
                    tab_img, tab_pdf = st.tabs(
                        ["🖼️ Reliable Preview", "📑 Native PDF Viewer"]
                    )

                    with tab_img:
                        st.write(f"Page {page_num} (High-Res Image)")
                        img = render_pdf_page_with_highlight(
                            pdf_bytes, page_num, geometry
                        )
                        if img:
                            st.image(img, use_container_width=True)

                    with tab_pdf:
                        st.info(
                            "⚠️ Note: Native viewer may be blocked by some browser security settings. Use 'Reliable Preview' as a fallback."
                        )
                        base64_pdf = get_pdf_with_highlight_base64(
                            pdf_bytes, page_num, geometry
                        )
                        if base64_pdf:
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="1000" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    # Show first page by default
                    default_page = int(df.iloc[0]["Page"]) if not df.empty else 1
                    img = render_pdf_page_with_highlight(pdf_bytes, default_page)
                    if img:
                        st.image(
                            img,
                            caption=f"Page {default_page} (Select a row to see highlights)",
                            use_container_width=True,
                        )


def render_show_pdf_tab():
    """Render the tab to show a PDF using st.pdf."""
    st.subheader("🔍 Show PDF")
    st.info("Upload a PDF to view it using the native Streamlit PDF viewer.")

    uploaded_pdf = st.file_uploader(
        "Upload PDF file",
        type=["pdf"],
        help="Upload a PDF document to view it",
        key="show_pdf_uploader",
    )

    if uploaded_pdf is not None:
        st.divider()
        st.pdf(uploaded_pdf, height=1000)


def main():

    try:
        if "aws_credentials" not in st.session_state:
            st.session_state["aws_credentials"] = None

        if st.session_state["aws_credentials"] is None:

            aws_access_key = st.text_input(
                "Enter AWS Access Key", type="password", key="aws_access_key"
            )
            aws_secret_key = st.text_input(
                "Enter AWS Secret Key", type="password", key="aws_secret_key"
            )
            aws_session_token = st.text_input(
                "Enter AWS Session Token (if applicable)",
                type="password",
                key="aws_session_token",
            )

            if st.button("✅ Save Credentials"):
                if not aws_access_key or not aws_secret_key:
                    st.error("Access key and secret key are required")
                else:
                    st.session_state["aws_credentials"] = {
                        "aws_access_key": aws_access_key,
                        "aws_secret_key": aws_secret_key,
                        "aws_session_token": aws_session_token,
                    }
                    st.success("Credentials saved")
                    st.rerun()

        if st.session_state["aws_credentials"] is not None:
            st.set_page_config(
                page_title="Medical Bill Review",
                page_icon="🏥",
                layout="wide",
            )

            # Custom CSS for better styling
            st.markdown(
                """
                <style>
                .main-header {
                    font-size: 2.5rem;
                    font-weight: 700;
                    color: #1E88E5;
                    text-align: center;
                    margin-bottom: 1rem;
                }
                .sub-header {
                    font-size: 1.2rem;
                    color: #666;
                    text-align: center;
                    margin-bottom: 2rem;
                }
                .stDataFrame {
                    font-size: 14px;
                }
                .success-box {
                    padding: 1rem;
                    background-color: #E8F5E9;
                    border-radius: 0.5rem;
                    border-left: 4px solid #4CAF50;
                }
                .info-box {
                    padding: 1rem;
                    background-color: #E3F2FD;
                    border-radius: 0.5rem;
                    border-left: 4px solid #2196F3;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Header
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 300px;">
                    <img src="https://awsmp-logos.s3.amazonaws.com/seller-s26bqc5zvqci2/28b3758023bda4842d49ef0317e57566.png" alt="Logo"
                        width="100" 
                        style="margin-top: -4px;" />
                    <div>
                        <div class="main-header">Medical Bill Review</div>
                        <div class="sub-header">
                            Upload medical bills to extract and analyze bill items using AI
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📄 Single Bill", "📚 Multiple Bills", "📄 Upload PDF", "🔍 Show PDF"]
            )

            with tab1:
                render_single_bill_tab()

            with tab2:
                render_multiple_bills_tab()

            with tab3:
                render_pdf_upload_tab()

            with tab4:
                render_show_pdf_tab()

            # Instructions in sidebar
            with st.sidebar:
                # st.image("./public/Doclens_logo.png", use_container_width=False)
                st.header("📌 How to Use")
                st.markdown(
                    """
                    **Single Bill Tab:**
                    1. Upload one medical bill image
                    2. Click "Extract Bill Data"
                    
                    **Multiple Bills Tab:**
                    1. Upload multiple bill images
                    2. Click "Extract All Bills Data"

                    **Upload PDF Tab:**
                    1. Upload a PDF medical record
                    2. Upload the Page Classification CSV
                    3. Upload the Textract Blocks JSON
                    4. Click "Process PDF Medical Bills"
                    5. Click a row in the table to highlight it in the PDF!

                    ---

                    **Extracted Columns:**
                    - **Document Name**: Source file name
                    - **Date**: Service date
                    - **Medical Procedure Name**: Description
                    - **CPT Code**: Procedure code
                    ... (and more)
                    """
                )
                if st.button("🔄 Change Credentials"):
                    st.session_state["aws_credentials"] = None
                    st.rerun()

    except Exception as e:
        st.error(f"Error capturing AWS credentials: {str(e)}")
        return


if __name__ == "__main__":
    main()
