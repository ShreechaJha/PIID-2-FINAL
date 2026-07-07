import json
import base64
import os
import re

notebook_path = r"c:\Users\user\Desktop\PIID PROJECT 2\DATA_ANALYSIS_PROJECT\DATA_ANALYSIS_PROJECT_FINAL_GITHUB.ipynb"
output_dir = r"c:\Users\user\Desktop\PIID PROJECT 2\DATA_ANALYSIS_PROJECT\visualizations"
report_path = r"c:\Users\user\Desktop\PIID PROJECT 2\DATA_ANALYSIS_PROJECT\Visualizations_Report.md"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

image_count = 0
last_markdown = ""
report_content = "# 📊 Sales Data Visualizations & Insights\n\nThis document contains all the key graphs and visualizations extracted from the analysis, along with their descriptions.\n\n"

for cell in nb.get('cells', []):
    cell_type = cell.get('cell_type')
    
    if cell_type == 'markdown':
        source = cell.get('source', [])
        # join and strip to get text
        text = "".join(source).strip()
        if text:
            # Clean up the markdown a bit
            last_markdown = text
            
    elif cell_type == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                image_count += 1
                img_filename = f"graph_{image_count}.png"
                img_path = os.path.join(output_dir, img_filename)
                
                # Save the image
                with open(img_path, "wb") as fh:
                    fh.write(base64.b64decode(img_data))
                
                # Add to report
                report_content += f"---\n\n"
                report_content += f"### Visualization {image_count}\n\n"
                
                # Add the last markdown context if it seems like a header or description
                if last_markdown:
                    # Just add the first few lines of the markdown if it's long, or the whole thing
                    report_content += f"{last_markdown}\n\n"
                
                report_content += f"![Visualization {image_count}](./visualizations/{img_filename})\n\n"
                
                # Clear last markdown so we don't repeat it for multiple plots unless intended, 
                # but usually each plot has its own cell.
                last_markdown = ""

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"Successfully extracted {image_count} images and generated Visualizations_Report.md.")
