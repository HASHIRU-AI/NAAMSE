import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io


def generate_pdf_report(report_data: Dict[str, Any], output_path: str = "naamse_report.pdf"):
    """
    Generate a professional PDF report from NAAMSE fuzzer results.

    Args:
        report_data: The report dictionary from generate_report_node
        output_path: Path where PDF will be saved
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Extract data
    summary = report_data.get("summary", {})
    all_prompts = report_data.get("all_prompts_with_scores_and_history", [])

    # Title Page
    story.append(Paragraph("NAAMSE Security Assessment Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    exec_summary = _generate_executive_summary(summary, all_prompts)
    story.append(Paragraph(exec_summary, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # Key Metrics Table
    story.append(Paragraph("Key Metrics", heading_style))
    metrics_table = _create_metrics_table(summary)
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # Risk Severity Breakdown
    story.append(Paragraph("Risk Severity Distribution", heading_style))
    severity_chart = _create_severity_chart(all_prompts)
    if severity_chart:
        story.append(severity_chart)
    story.append(Spacer(1, 0.3*inch))

    # Page Break
    story.append(PageBreak())

    # Iteration Progression Chart
    story.append(Paragraph("Fuzzing Effectiveness Over Time", heading_style))
    iteration_chart = _create_iteration_chart(
        summary.get("iteration_progression", []))
    if iteration_chart:
        story.append(iteration_chart)
    story.append(Spacer(1, 0.3*inch))

    # Mutation Effectiveness Chart
    story.append(Paragraph("Mutation Type Effectiveness", heading_style))
    mutation_chart = _create_mutation_chart(summary.get("mutation_report", []))
    if mutation_chart:
        story.append(mutation_chart)
    story.append(Spacer(1, 0.3*inch))

    # Mutation Radar Chart
    story.append(Paragraph("Mutation Type Radar Analysis", heading_style))
    mutation_radar = _create_radar_chart(
        summary.get("mutation_report", []), "Mutation Types")
    if mutation_radar:
        story.append(mutation_radar)
    story.append(Spacer(1, 0.3*inch))

    # Page Break
    story.append(PageBreak())

    # Top Vulnerabilities
    story.append(Paragraph("Top 10 Vulnerabilities", heading_style))
    top_vulns = _create_top_vulnerabilities_table(all_prompts)
    story.append(top_vulns)
    story.append(Spacer(1, 0.3*inch))

    # Cluster Analysis
    story.append(PageBreak())
    story.append(Paragraph("Attack Vector Analysis by Cluster", heading_style))
    cluster_table = _create_cluster_table(summary.get("cluster_report", []))
    story.append(cluster_table)
    story.append(Spacer(1, 0.3*inch))

    # Cluster Radar Chart
    story.append(Paragraph("Cluster Radar Analysis", heading_style))
    cluster_radar = _create_radar_chart(
        summary.get("cluster_report", []), "Clusters")
    if cluster_radar:
        story.append(cluster_radar)
    story.append(Spacer(1, 0.3*inch))

    # Build PDF
    doc.build(story)
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
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
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
    filtered_data = [(size, label, color) for size, label, color in data if size > 0]
    
    if not filtered_data:
        return None
    
    sizes = [d[0] for d in filtered_data]
    labels = [d[1] for d in filtered_data]
    colors_list = [d[2] for d in filtered_data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(sizes, labels=labels, colors=colors_list,
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Risk Severity Distribution')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=5*inch, height=3.3*inch)


def _create_iteration_chart(iteration_data: List[Dict]) -> Image:
    """Create iteration progression line chart."""
    if not iteration_data:
        return None

    iterations = [d["iteration"] for d in iteration_data]
    avg_scores = [d["avg_score"] for d in iteration_data]
    max_scores = [d["max_score"] for d in iteration_data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, avg_scores, marker='o',
            label='Average Score', linewidth=2)
    ax.plot(iterations, max_scores, marker='s', label='Max Score', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Fuzzing Effectiveness Over Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=6*inch, height=3.5*inch)


def _create_mutation_chart(mutation_data: List[Dict]) -> Image:
    """Create mutation effectiveness bar chart."""
    if not mutation_data:
        return None

    # Take top 10 mutations
    top_mutations = mutation_data[:10]
    mutation_types = [d["mutation_type"] for d in top_mutations]
    avg_scores = [d["avg_score"] for d in top_mutations]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(mutation_types, avg_scores, color='#3498db')
    ax.set_xlabel('Average Score')
    ax.set_title('Top 10 Mutation Types by Effectiveness')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                ha='left', va='center', fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=6*inch, height=4*inch)


def _create_radar_chart(data: List[Dict], chart_type: str) -> Image:
    """Create radar/star chart for mutation types or clusters."""
    if not data or len(data) < 3:
        return None

    # Take top 8 items for readability
    top_items = data[:8]

    if chart_type == "Mutation Types":
        labels = [d["mutation_type"][:20]
                  for d in top_items]  # Truncate long names
        values = [d["avg_score"] for d in top_items]
    else:  # Clusters
        labels = [d["cluster"][:20] for d in top_items]
        values = [d["avg_score"] for d in top_items]

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    values += values[:1]  # Complete the circle
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
    ax.set_title(f'{chart_type} - Average Score Distribution', size=12, pad=20)
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image(buf, width=5.5*inch, height=5.5*inch)


def _create_top_vulnerabilities_table(all_prompts: List[Dict]) -> Table:
    """Create table of top 10 vulnerabilities."""
    # Sort by score and take top 10
    sorted_prompts = sorted(all_prompts, key=lambda x: x.get(
        "score", 0), reverse=True)[:10]

    data = [["Rank", "Score", "Prompt Preview", "Mutation Type"]]

    for i, prompt in enumerate(sorted_prompts, 1):
        score = prompt.get("score", 0)
        # Handle prompt that might be list of strings or list of dicts
        prompt_parts = prompt.get("prompt", [])
        if prompt_parts and isinstance(prompt_parts[0], dict):
            prompt_text = str(prompt_parts[0])[:100] + "..."
        else:
            prompt_text = " ".join(str(p) for p in prompt_parts)[:100] + "..."
        mutation_type = prompt.get("metadata", {}).get(
            "mutation_type", "unknown")

        data.append([
            str(i),
            f"{score:.1f}",
            prompt_text,
            mutation_type
        ])

    table = Table(data, colWidths=[0.5*inch, 0.7*inch, 3.5*inch, 1.3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return table


def _create_cluster_table(cluster_data: List[Dict]) -> Table:
    """Create cluster analysis table."""
    if not cluster_data:
        return Table([["No cluster data available"]])

    # Take top 15 clusters
    top_clusters = cluster_data[:15]

    data = [["Cluster", "Count", "Avg Score", "Max Score", "Description"]]

    for cluster in top_clusters:
        data.append([
            cluster.get("cluster", "N/A"),
            str(cluster.get("count", 0)),
            f"{cluster.get('avg_score', 0):.1f}",
            f"{cluster.get('max_score', 0):.1f}",
            cluster.get("description", "")[:60] + "..."
        ])

    table = Table(data, colWidths=[1*inch, 0.6 *
                  inch, 0.8*inch, 0.8*inch, 2.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return table


if __name__ == "__main__":
    # Test with example data
    with open("tests/data/final_report.json", "r", encoding="utf-8") as f:
        test_state = json.load(f)

    generate_pdf_report(test_state["report"], "test_naamse_report.pdf")
