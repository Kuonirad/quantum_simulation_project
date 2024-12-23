import logging
import os
import subprocess

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class SimulationResult:
    def __init__(self):
        self.equation = "$E = mc^2$"

    def plot_data(self, path):
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.savefig(path)
        plt.close()


# Configure Logging
logging.basicConfig(
    filename="scientific_paper_generation.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


def convert_latex_to_png(latex_code, output_path):
    """
    Converts LaTeX code to a PNG image using LaTeX and dvipng.
    """
    try:
        # Create temporary LaTeX file
        tex_content = f"""
        \\documentclass{{standalone}}
        \\usepackage{{amsmath}}
        \\begin{{document}}
        {latex_code}
        \\end{{document}}
        """
        tex_dir = os.path.dirname(output_path)
        tex_file = os.path.join(tex_dir, "temp_eq.tex")
        with open(tex_file, "w") as f:
            f.write(tex_content)
        logging.debug(f"Created LaTeX file at {tex_file}")

        # Compile LaTeX to DVI
        subprocess.run(["latex", "-interaction=nonstopmode", tex_file], check=True, cwd=tex_dir)
        logging.debug("LaTeX compilation successful.")

        # Convert DVI to PNG
        dvi_file = tex_file.replace(".tex", ".dvi")
        subprocess.run(["dvipng", "-D", "300", "-T", "tight", "-o", output_path, dvi_file], check=True, cwd=tex_dir)
        logging.debug(f"Converted DVI to PNG at {output_path}")

        # Clean up auxiliary files
        for ext in [".aux", ".log", ".dvi", ".tex"]:
            aux_file = tex_file.replace(".tex", ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)
                logging.debug(f"Removed auxiliary file: {aux_file}")

        logging.info(f"Successfully converted LaTeX to PNG: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"LaTeX conversion failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during LaTeX conversion: {e}")
        return False


def embed_content(pdf_path, images, equations, citations):
    """
    Embeds images, equations, and citations into a PDF.
    """
    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        logging.info(f"Creating PDF: {pdf_path}")
        y_position = height - 50  # Starting y position

        # Embed Images
        for idx, image_path in enumerate(images):
            try:
                img = ImageReader(image_path)
                c.drawImage(img, x=50, y=y_position, width=200, height=150)
                logging.debug(f"Embedded image {idx + 1}: {image_path} at position (50, {y_position})")
                y_position -= 170  # Adjust y position for next image
                if y_position < 100:
                    c.showPage()
                    y_position = height - 50
            except Exception as e:
                logging.error(f"Error embedding image {image_path}: {e}")

        # Embed Equations
        for idx, equation_path in enumerate(equations):
            try:
                eq_img = ImageReader(equation_path)
                c.drawImage(eq_img, x=300, y=y_position, width=200, height=150)
                logging.debug(f"Embedded equation {idx + 1}: {equation_path} at position (300, {y_position})")
                y_position -= 170
                if y_position < 100:
                    c.showPage()
                    y_position = height - 50
            except Exception as e:
                logging.error(f"Error embedding equation {equation_path}: {e}")

        # Embed Citations as Text
        c.setFont("Helvetica", 10)
        y_position = 50  # Bottom of the page
        for idx, citation in enumerate(citations):
            try:
                c.drawString(50, y_position, f"{idx + 1}. {citation}")
                logging.debug(f"Embedded citation {idx + 1}: {citation} at position (50, {y_position})")
                y_position += 15
                if y_position > height - 50:
                    c.showPage()
                    y_position = 50
            except Exception as e:
                logging.error(f"Error embedding citation {citation}: {e}")

        c.save()
        logging.info(f"PDF generation completed: {pdf_path}")
        return True
    except Exception as e:
        logging.error(f"Error during PDF embedding: {e}")
        return False


def verify_pdf_content(pdf_path, expected_images, expected_equations):
    """
    Verifies that the PDF contains the expected number of images and equations.
    """
    try:
        doc = fitz.open(pdf_path)
        actual_images = 0
        actual_equations = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            actual_images += len(images)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_name = base_image.get("name", "").lower()
                # Adjust this condition based on how equations are named
                if "equation" in image_name or "eq_" in image_name:
                    actual_equations += 1
        doc.close()

        logging.info("Verification Results:")
        logging.info(f"Expected Images: {expected_images}, Actual Images: {actual_images}")
        logging.info(f"Expected Equations: {expected_equations}, Actual Equations: {actual_equations}")

        verification_passed = True
        if actual_images < expected_images:
            logging.error(f"Missing images. Expected: {expected_images}, Found: {actual_images}")
            verification_passed = False
        if actual_equations < expected_equations:
            logging.error(f"Missing equations. Expected: {expected_equations}, Found: {actual_equations}")
            verification_passed = False

        if verification_passed:
            logging.info("Content verification passed.")
        else:
            logging.error("Content verification failed.")

        return verification_passed
    except Exception as e:
        logging.error(f"Error during PDF verification: {e}")
        return False


def generate_scientific_paper(simulation_results, output_path="final_scientific_paper.pdf"):
    """Generates a scientific paper based on the quantum simulation results."""
    logging.info("Starting scientific paper generation")

    images = []
    equations = []
    for idx, result in enumerate(simulation_results):
        plot_path = f"/home/ubuntu/images/plot_{idx}.png"
        result.plot_data(plot_path)
        images.append(plot_path)

        eq_path = f"/home/ubuntu/equations/eq_{idx}.png"
        convert_latex_to_png(result.equation, eq_path)
        equations.append(eq_path)

    citations = [
        "[1] Russell, W., The Universal One, 1926.",
        "[2] Bohm, D., Wholeness and the Implicate Order, 1980.",
        "[3] Wilber, K., Integral Psychology, 2000.",
    ]

    embed_success = embed_content(output_path, images, equations, citations)
    verification_passed = verify_pdf_content(output_path, len(images), len(equations))

    if embed_success and verification_passed:
        logging.info(f"Scientific paper generated successfully: {output_path}")
        return True
    else:
        logging.error("Failed to generate scientific paper.")
        return False


if __name__ == "__main__":
    # This will be replaced with actual simulation results
    class SimulationResult:
        def __init__(self):
            self.equation = "$E = mc^2$"

        def plot_data(self, path):
            # Dummy plot function
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot([1, 2, 3], [1, 4, 9])
            plt.savefig(path)
            plt.close()

    dummy_results = [SimulationResult() for _ in range(5)]
    generate_scientific_paper(dummy_results)
