from fpdf import FPDF
import csv

def generate_pdf(score, consistency, heatmap, out="deepfake_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Deepfake Detection Report", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, f"Deepfake Probability: {score}%", ln=True)
    pdf.cell(200, 10, f"Audio-Visual Consistency: {consistency}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, "Heatmap:", ln=True)
    pdf.image(heatmap, x=10, w=180)

    pdf.output(out)
    return out


def generate_csv(score, consistency, out="deepfake_report.csv"):
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Deepfake Probability (%)", score])
        writer.writerow(["Audio-visual Consistency", consistency])
    return out
