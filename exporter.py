import os
import shutil
from glob import glob

import nbformat
from nbconvert import HTMLExporter


def _create_merged_presentation_file(file_paths, output_file):
    html_start = "<!DOCTYPE html>\n<html>\n<head>\n<title>Presentation</title>\n</head>\n<body>\n"
    html_end = "</body>\n</html>"

    combined_content = html_start
    combined_content_with_code = html_start

    for file_path in file_paths:
        if "Append" in file_path:
            break
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            combined_content += f"<div class='file-content'>\n{content}\n</div>\n"

        with open(file_path.replace(".html", "_with_code.html"), "r", encoding="utf-8") as file:
            content = file.read()
            combined_content_with_code += f"<div class='file-content'>\n{content}\n</div>\n"

    combined_content += html_end

    with open(output_file, "w", encoding="utf-8") as output:
        output.write(combined_content)

    with open(output_file.replace(".html", "_with_code.html"), "w", encoding="utf-8") as output:
        output.write(combined_content_with_code)


def _export_html(notebook_path, html_output_path, exclude_input):
    custom_css = """
    <style>
    body { background-color: #F0F0F0 !important; }
    [data-mime-type='application/vnd.jupyter.stderr'] { display: none; }
    main {
        display: flex;
        flex-direction: column; /* Stack children vertically */
        align-items: center; /* Center children horizontally */
    }
    div.jp-RenderedImage img {
        max-width: 960px; /* Set maximum width */
        height: auto; /* Maintain aspect ratio */
    }
    .jp-MarkdownCell {
    }
    .jp-Cell{
        max-width: 1200px !important;
        height: auto;

    }
    </style>
    """

    exporter = HTMLExporter()
    exporter.exclude_input = exclude_input

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

        html, _ = exporter.from_notebook_node(nb)

    # if not exclude_input:
    #     html_output_path = html_output_path.replace(".html", "_with_code.html")

    html = html.replace("</head>", f"{custom_css}</head>")

    return html, html_output_path


def _generate_table_of_contents(target_export_path: str):
    """
    Generates an HTML table of contents for the exported notebooks, organizing them by top-level section number.
    Each notebook is listed once with links to both its versions. Section titles use the name from the X.0 notebook
    or "PLACEHOLDER" if no such notebook exists.

    Parameters:
    - target_export_path (str): The path where the HTML files are stored.
    """

    full_presentation_path = f"presentation.html"
    toc = {}
    for root, _, files in os.walk(target_export_path):
        for file in files:
            if file.endswith(".html"):
                stripped_name = file.replace("_with_code.html", "").replace(".html", "")
                section = stripped_name.split("_")[0].split(".")[
                    0
                ]  # Extract top-level section number
                if section not in toc:
                    toc[section] = {}
                file_path = os.path.join(root, file)
                if "_with_code" in file:
                    toc[section][stripped_name] = toc[section].get(stripped_name, {})
                    toc[section][stripped_name]["with_code"] = file_path
                else:
                    toc[section][stripped_name] = toc[section].get(stripped_name, {})
                    toc[section][stripped_name]["default"] = file_path

    section_titles = {}
    for section in toc.keys():
        found = False
        for nb in toc[section]:
            if nb.startswith(f"{section}.0"):
                title = nb.split("_", 1)[1].replace("_", " ")
                section_titles[section] = title
                found = True
                break
        if not found:
            section_titles[section] = "PLACEHOLDER"

    file_paths = []
    with open(
            os.path.join(target_export_path, "index.html"), "w", encoding="utf-8"
    ) as f:
        f.write(
            "<html><body><h1>Lenders Club Loan Dataset Loan Risk Prediction:</h1>\n"
        )
        f.write(
            f'<p><a href="{full_presentation_path}">Full Presentation</a></p><br>\n'
        )
        for section, notebooks in sorted(toc.items()):
            f.write(f"<h2>{section} {section_titles[section]}</h2>\n<ul>\n")
            for base_name, versions in sorted(notebooks.items()):
                default_link = os.path.relpath(
                    versions.get("default", ""), target_export_path
                ).replace(os.path.sep, "/")

                file_paths.append(f"{target_export_path}/{default_link}")
                with_code_link = os.path.relpath(
                    versions.get("with_code", ""), target_export_path
                ).replace(os.path.sep, "/")
                f.write(
                    f'<li><a href="{default_link}">{base_name}</a> (<a href="{with_code_link}">with code</a>)</li>\n'
                )
            f.write("</ul>\n")
        f.write("</body></html>")

    _create_merged_presentation_file(
        file_paths, f"{target_export_path}/presentation.html"
    )


def export_nested_notebooks(notebooks_root_path: str, target_export_path: str):
    for root, _, files in os.walk(notebooks_root_path):
        for file in files:
            if file.endswith(".ipynb") and file[0].isdigit():
                notebook_path = os.path.join(root, file)
                relative_path = os.path.relpath(notebook_path, notebooks_root_path)
                html_output_path = os.path.join(
                    target_export_path, relative_path
                ).replace(".ipynb", ".html")

                os.makedirs(os.path.dirname(html_output_path), exist_ok=True)

                for exclude_input in [True, False]:
                    modified_html_output_path = (
                        html_output_path.replace(".html", "_with_code.html")
                        if not exclude_input
                        else html_output_path
                    )

                    try:
                        html, final_html_output_path = _export_html(
                            notebook_path, modified_html_output_path, exclude_input
                        )

                        with open(final_html_output_path, "w", encoding="utf-8") as f:
                            f.write(html)
                    except Exception as e:
                        print(
                            f"Failed exporting :{modified_html_output_path}:\n{e}"
                        )  # with:\n{e}")

    _generate_table_of_contents(target_export_path)
