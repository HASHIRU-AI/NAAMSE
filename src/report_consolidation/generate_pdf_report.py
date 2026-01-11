import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
import io


# Global table style
def _get_modern_table_style(header_font_size=10, body_font_size=8):
    """Return consistent modern table styling."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), header_font_size),
        ('FONTSIZE', (0, 1), (-1, -1), body_font_size),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#3498db')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])


# Global cell style for wrapping text
def _get_table_cell_style(font_size=8):
    """Return paragraph style for table cells with text wrapping."""
    return ParagraphStyle(
        'TableCellStyle',
        parent=getSampleStyleSheet()['BodyText'],
        fontSize=font_size,
        leading=font_size + 2
    )


def generate_pdf_report(report_data: Dict[str, Any], output_path: str = "naamse_report.pdf"):
    """
    Generate a professional PDF report from NAAMSE fuzzer results.

    Args:
        report_data: The report dictionary from generate_report_node
        output_path: Path where PDF will be saved
    """

    # Generate timestamp once
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def add_header(canvas, doc):
        """Add header with timestamp to each page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#7f8c8d'))
        canvas.drawRightString(
            letter[0] - 0.25*inch, letter[1] - 0.5*inch, f"Generated: {generated_time}")
        canvas.restoreState()

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=16,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=0,
        leftIndent=0
    )

    # Extract data
    summary = report_data.get("summary", {})
    all_prompts = report_data.get("all_prompts_with_scores_and_history", [])

    # Title Page
    story.append(Paragraph("NAAMSE Security Assessment Report", title_style))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    exec_summary = _generate_executive_summary(summary, all_prompts)
    story.append(Paragraph(exec_summary, styles['BodyText']))

    # Key Metrics Table
    story.append(Paragraph("Key Metrics", heading_style))
    metrics_table = _create_metrics_table(summary)
    story.append(metrics_table)

    # Page Break
    story.append(PageBreak())

    # Risk Severity Breakdown
    severity_section = []
    severity_section.append(
        Paragraph("Risk Severity Distribution", heading_style))
    severity_chart = _create_severity_chart(all_prompts)
    if severity_chart:
        severity_section.append(severity_chart)
    story.append(KeepTogether(severity_section))

    # Iteration Progression Chart
    iteration_section = []
    iteration_section.append(
        Paragraph("Attack Effectiveness Over Time", heading_style))
    iteration_chart = _create_iteration_chart(
        summary.get("iteration_progression", []))
    if iteration_chart:
        iteration_section.append(iteration_chart)
    story.append(KeepTogether(iteration_section))

    # Page Break
    story.append(PageBreak())

    # Top Vulnerabilities
    story.append(Paragraph("Top 10 Vulnerabilities", heading_style))
    top_vulns = _create_top_vulnerabilities_table(all_prompts)
    story.append(top_vulns)

    # Page Break
    story.append(PageBreak())

    # Cluster Analysis
    cluster_section = []
    cluster_section.append(
        Paragraph("Attack Vector Analysis by Attack type", heading_style))
    cluster_table = _create_cluster_table(summary.get("cluster_report", []))
    cluster_section.append(cluster_table)
    story.append(KeepTogether(cluster_section))

    # Cluster Radar Chart
    cluster_radar_section = []
    cluster_radar_section.append(
        Paragraph("Attack type Radar Analysis", heading_style))
    cluster_radar = _create_radar_chart(
        summary.get("cluster_report", []), "Clusters")
    if cluster_radar:
        cluster_radar_section.append(cluster_radar)
    story.append(KeepTogether(cluster_radar_section))

    # Page Break
    story.append(PageBreak())

    # Mutation Type Analysis Table
    mutation_section = []
    mutation_section.append(Paragraph("Mutation Type Analysis", heading_style))
    mutation_table = _create_mutation_table(summary.get("mutation_report", []))
    mutation_section.append(mutation_table)
    story.append(KeepTogether(mutation_section))

    # Mutation Radar Chart
    mutation_radar_section = []
    mutation_radar_section.append(
        Paragraph("Mutation Type Radar Analysis", heading_style))
    mutation_radar = _create_radar_chart(
        summary.get("mutation_report", []), "Mutation Types")
    if mutation_radar:
        mutation_radar_section.append(mutation_radar)
    story.append(KeepTogether(mutation_radar_section))

    # Page Break
    story.append(PageBreak())

    # Build PDF with header
    doc.build(story, onFirstPage=add_header, onLaterPages=add_header)
    print(f"PDF report generated: {output_path}")


def _generate_executive_summary(summary: Dict[str, Any], all_prompts: List[Dict]) -> str:
    """Generate executive summary text."""
    total = summary.get("total_prompts_tested", 0)
    max_score = summary.get("max_score", 0)
    avg_score = summary.get("avg_score", 0)
    high_count = summary.get("high_score_count", 0)

    # Calculate risk level
    if max_score >= 80:
        risk_level = "CRITICAL"
    elif max_score >= 60:
        risk_level = "HIGH"
    elif max_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Calculate improvement
    iterations = summary.get("iteration_progression", [])
    improvement = "N/A"
    if len(iterations) >= 2:
        first_avg = iterations[0].get("avg_score", 0)
        last_avg = iterations[-1].get("avg_score", 0)
        if first_avg > 0:
            improvement = f"{((last_avg - first_avg) / first_avg * 100):.1f}%"

    summary_text = f"""
    This security assessment tested {total} adversarial prompts against the target AI agent. 
    The maximum vulnerability score achieved was {max_score:.1f}/100, indicating a <b>{risk_level}</b> risk level. 
    {high_count} prompts ({(high_count/total*100) if total else 0:.1f}%) exceeded the vulnerability threshold, 
    with an average score of {avg_score:.1f}/100. Fuzzing effectiveness improved by {improvement} across iterations.
    """
    return summary_text.strip()


def _create_metrics_table(summary: Dict[str, Any]) -> Table:
    """Create key metrics summary table."""
    data = [
        ["Metric", "Value"],
        ["Total Prompts Tested", str(summary.get("total_prompts_tested", 0))],
        ["Maximum Score", f"{summary.get('max_score', 0):.1f}/100"],
        ["Average Score", f"{summary.get('avg_score', 0):.1f}/100"],
        ["High-Risk Prompts", str(summary.get("high_score_count", 0))],
        ["Unique Clusters", str(len(summary.get("cluster_report", [])))],
        ["Mutation Types", str(len(summary.get("mutation_report", [])))],
    ]

    table = Table(data, colWidths=[3*inch, 2*inch])
    table.setStyle(_get_modern_table_style(
        header_font_size=13, body_font_size=11))
    return table


def _create_severity_chart(all_prompts: List[Dict]) -> Image:
    """Create risk severity distribution pie chart."""
    if not all_prompts:
        return None

    # Categorize by severity
    critical = sum(1 for p in all_prompts if p.get("score", 0) >= 80)
    high = sum(1 for p in all_prompts if 60 <= p.get("score", 0) < 80)
    medium = sum(1 for p in all_prompts if 40 <= p.get("score", 0) < 60)
    low = sum(1 for p in all_prompts if p.get("score", 0) < 40)

    # Filter out zero values
    data = [
        (critical, f'Critical (≥80): {critical}', '#e74c3c'),
        (high, f'High (60-79): {high}', '#e67e22'),
        (medium, f'Medium (40-59): {medium}', '#f39c12'),
        (low, f'Low (<40): {low}', '#27ae60')
    ]

    # Only include non-zero categories
    filtered_data = [(size, label, color)
                     for size, label, color in data if size > 0]

    if not filtered_data:
        return None

    sizes = [d[0] for d in filtered_data]
    labels = [d[1] for d in filtered_data]
    colors_list = [d[2] for d in filtered_data]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list,
                                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10},
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.axis('equal')
    # plt.title('Risk Severity Distribution',
    #           fontsize=14, fontweight='bold', pad=20)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=6*2/3*inch, height=5*2/3*inch, kind='proportional')


def _create_iteration_chart(iteration_data: List[Dict]) -> Image:
    """Create iteration progression line chart."""
    if not iteration_data:
        return None

    iterations = [d["iteration"] for d in iteration_data]
    avg_scores = [d["avg_score"] for d in iteration_data]
    max_scores = [d["max_score"] for d in iteration_data]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    ax.plot(iterations, avg_scores, marker='o',
            label='Average Score', linewidth=2.5, color='#3498db', markersize=8)
    ax.plot(iterations, max_scores, marker='s', label='Max Score', linewidth=2.5,
            color='#e74c3c', markersize=8)
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    # ax.set_title('Fuzzing Effectiveness Over Iterations',
    #              fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=6*2/3*inch, height=5*2/3*inch, kind='proportional')


def _create_mutation_chart(mutation_data: List[Dict]) -> Image:
    """Create mutation effectiveness bar chart."""
    if not mutation_data:
        return None

    # Take top 10 mutations
    top_mutations = mutation_data[:10]
    mutation_types = [d["mutation_type"] for d in top_mutations]
    avg_scores = [d["avg_score"] for d in top_mutations]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    bars = ax.barh(mutation_types, avg_scores, color='#3498db', edgecolor='#2980b9',
                   linewidth=1.5, height=0.7)
    ax.set_xlabel('Average Score', fontsize=12, fontweight='bold')
    # ax.set_title('Top 10 Mutation Types by Effectiveness',
    #              fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=10*2/3*inch, height=5*2/3*inch, kind='proportional')


def _create_attack_type_chart(cluster_data: List[Dict]) -> Image:
    """Create attack type effectiveness bar chart."""
    if not cluster_data:
        return None

    # Take top 10 clusters
    top_clusters = cluster_data[:10]
    cluster_labels = [d["cluster"][:30] for d in top_clusters]
    avg_scores = [d["avg_score"] for d in top_clusters]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    bars = ax.barh(cluster_labels, avg_scores, color='#e74c3c', edgecolor='#c0392b',
                   linewidth=1.5, height=0.7)
    ax.set_xlabel('Average Score', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=10*2/3*inch, height=5*2/3*inch, kind='proportional')


def _create_radar_chart(data: List[Dict], chart_type: str) -> Image:
    """Create radar/star chart for mutation types or clusters."""
    if not data or len(data) < 3:
        return None

    # Take top 8 items for readability
    top_items = data[:8]

    if chart_type == "Mutation Types":
        labels = [d["mutation_type"]
                  for d in top_items]  # Truncate long names
        values = [d["avg_score"] for d in top_items]
    else:  # Clusters
        labels = [d["cluster"] for d in top_items]
        values = [d["avg_score"] for d in top_items]

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    values += values[:1]  # Complete the circle
    angles += angles[:1]

    # Create plot with square aspect ratio
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2.5, color='#3498db', markersize=8)
    ax.fill(angles, values, alpha=0.2, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9, color='gray')

    ax.grid(True, alpha=0.3, linestyle='--')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close()

    return Image(buf, width=5*inch, height=5*inch, kind='proportional')


def _create_top_vulnerabilities_table(all_prompts: List[Dict]) -> Table:
    """Create table of top 10 vulnerabilities."""
    # Sort by score and take top 10
    sorted_prompts = sorted(all_prompts, key=lambda x: x.get(
        "score", 0), reverse=True)[:10]

    cell_style = _get_table_cell_style(font_size=8)

    data = [["Rank", "Score", "Attack Type", "Mutation Type", "Prompt Preview"]]

    for i, prompt in enumerate(sorted_prompts, 1):
        score = prompt.get("score", 0)
        # Handle prompt that might be list of strings or list of dicts
        prompt_parts = prompt.get("prompt", [])
        prompt_text = " ".join(str(p) for p in prompt_parts)[:250] + "..."

        mutation_type = prompt.get("metadata", {}).get(
            "mutation_type", "unknown")
        cluster_info = prompt.get("metadata", {}).get("cluster_info", {})
        attack_type = cluster_info.get(
            "label", cluster_info.get("cluster_label", "Unknown"))

        data.append([
            str(i),
            f"{score:.1f}",
            Paragraph(attack_type, cell_style),
            Paragraph(mutation_type, cell_style),
            Paragraph(prompt_text, cell_style)
        ])

    table = Table(data, colWidths=[
                  0.5*inch, 0.5*inch, 1.2*inch, 1.2*inch, 4.5*inch])
    style = _get_modern_table_style(header_font_size=10, body_font_size=8)
    style.add('VALIGN', (0, 1), (-1, -1), 'TOP')
    table.setStyle(style)
    return table


def _create_cluster_table(cluster_data: List[Dict]) -> Table:
    """Create cluster analysis table."""
    if not cluster_data:
        return Table([["No cluster data available"]])

    # Take top 15 clusters
    top_clusters = cluster_data[:15]

    data = [["Cluster", "Count", "Avg Score", "Max Score", "Description"]]

    cell_style = _get_table_cell_style(font_size=8)

    for cluster in top_clusters:
        data.append([
            Paragraph(cluster.get("cluster", "N/A"), cell_style),
            str(cluster.get("count", 0)),
            f"{cluster.get('avg_score', 0):.1f}",
            f"{cluster.get('max_score', 0):.1f}",
            Paragraph(cluster.get("description", ""), cell_style)
        ])

    table = Table(data, colWidths=[1.2*inch, 0.6 *
                  inch, 0.8*inch, 0.8*inch, 2.8*inch])
    table.setStyle(_get_modern_table_style(
        header_font_size=9, body_font_size=8))
    return table


def _create_mutation_table(mutation_data: List[Dict]) -> Table:
    """Create mutation type analysis table."""
    if not mutation_data:
        return Table([["No mutation data available"]])

    # Take top 15 mutation types
    top_mutations = mutation_data[:15]

    data = [["Mutation Type", "Count", "Avg Score", "Max Score"]]

    cell_style = _get_table_cell_style(font_size=8)

    for mutation in top_mutations:
        data.append([
            Paragraph(mutation.get("mutation_type", "N/A"), cell_style),
            str(mutation.get("count", 0)),
            f"{mutation.get('avg_score', 0):.1f}",
            f"{mutation.get('max_score', 0):.1f}"
        ])

    table = Table(data, colWidths=[3*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    table.setStyle(_get_modern_table_style(
        header_font_size=9, body_font_size=8))
    return table


if __name__ == "__main__":
    # Test with example data
    with open("tests/data/final_report_2.json", "r", encoding="utf-8") as f:
        test_state = json.load(f)

    generate_pdf_report(test_state["report"], "test_naamse_report.pdf")
