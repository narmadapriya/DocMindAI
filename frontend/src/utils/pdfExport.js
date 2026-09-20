function safeText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function addWrappedText(pdf, text, x, y, width, lineHeight = 6) {
  const lines = pdf.splitTextToSize(safeText(text), width);
  for (const line of lines) {
    if (y > 280) {
      pdf.addPage();
      y = 18;
    }
    pdf.text(line, x, y);
    y += lineHeight;
  }
  return y;
}

export async function downloadSummaryPdf({
  title = "DocMind AI Summary",
  filename = "docmind-summary.pdf",
  summary,
  citations = [],
}) {
  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF({ unit: "mm", format: "a4" });
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(16);
  pdf.text(title, 16, 18);

  pdf.setFont("helvetica", "normal");
  pdf.setFontSize(10);
  let y = 30;
  y = addWrappedText(pdf, summary || "No summary returned.", 16, y, 178, 5.5);

  if (citations.length) {
    y += 6;
    if (y > 270) {
      pdf.addPage();
      y = 18;
    }
    pdf.setFont("helvetica", "bold");
    pdf.text("Sources / Citations", 16, y);
    y += 7;
    pdf.setFont("helvetica", "normal");

    citations.forEach((citation, index) => {
      const line =
        typeof citation === "string"
          ? citation
          : citation.label ||
            citation.filename ||
            citation.quoted_text ||
            `Source ${index + 1}`;
      y = addWrappedText(pdf, `${index + 1}. ${line}`, 16, y, 178, 5);
    });
  }

  pdf.save(filename);
}

export async function downloadComparisonPdf({
  title = "DocMind AI Cross-Document Comparison",
  filename = "docmind-comparison.pdf",
  documentNames = [],
  rows = [],
  answer = "",
  citations = [],
}) {
  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF({ unit: "mm", format: "a4", orientation: "landscape" });
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(15);
  pdf.text(title, 14, 16);

  pdf.setFontSize(9);
  pdf.setFont("helvetica", "normal");
  if (documentNames.length) {
    pdf.text(`Documents: ${documentNames.join(" vs ")}`, 14, 24);
  }

  let y = 34;
  pdf.setFont("helvetica", "bold");
  pdf.text("Metric", 14, y);
  pdf.text(documentNames[0] || "Document A", 62, y);
  pdf.text(documentNames[1] || "Document B", 142, y);
  pdf.text("Difference", 220, y);
  y += 3;
  pdf.line(14, y, 282, y);
  y += 6;

  pdf.setFont("helvetica", "normal");
  rows.forEach((row) => {
    if (y > 190) {
      pdf.addPage();
      y = 18;
    }
    const values = [
      safeText(row.metric),
      safeText(row.document_a),
      safeText(row.document_b),
      safeText(row.change),
    ];
    const xs = [14, 62, 142, 220];
    const widths = [44, 74, 74, 60];
    const split = values.map((value, i) =>
      pdf.splitTextToSize(value || "N/A", widths[i]),
    );
    const maxLines = Math.max(...split.map((lines) => lines.length));
    split.forEach((lines, i) => pdf.text(lines, xs[i], y));
    y += Math.max(7, maxLines * 5 + 2);
    pdf.setDrawColor(50, 60, 80);
    pdf.line(14, y - 2, 282, y - 2);
  });

  if (answer) {
    y += 5;
    pdf.setFont("helvetica", "bold");
    pdf.text("Backend Comparison Response", 14, y);
    y += 6;
    pdf.setFont("helvetica", "normal");
    y = addWrappedText(pdf, answer, 14, y, 268, 5);
  }

  if (citations.length) {
    y += 5;
    if (y > 188) {
      pdf.addPage();
      y = 18;
    }
    pdf.setFont("helvetica", "bold");
    pdf.text("Sources / Citations", 14, y);
    y += 6;
    pdf.setFont("helvetica", "normal");
    citations.forEach((citation, index) => {
      const line =
        typeof citation === "string"
          ? citation
          : citation.label ||
            citation.filename ||
            citation.quoted_text ||
            `Source ${index + 1}`;
      y = addWrappedText(pdf, `${index + 1}. ${line}`, 14, y, 268, 5);
    });
  }

  pdf.save(filename);
}


export async function downloadConversationPdf({
  title = "DocMind AI Conversation",
  filename = "docmind-conversation.pdf",
  documentName = "",
  messages = [],
}) {
  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF({ unit: "mm", format: "a4" });
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(16);
  pdf.text(title, 16, 18);

  pdf.setFont("helvetica", "normal");
  pdf.setFontSize(9);
  let y = 27;

  if (documentName) {
    pdf.text(`Active document: ${documentName}`, 16, y);
    y += 8;
  }

  messages.forEach((message, index) => {
    if (y > 270) {
      pdf.addPage();
      y = 18;
    }

    const role = message.role === "user" ? "You" : "DocMind AI";
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(10);
    pdf.text(`${role}:`, 16, y);
    y += 6;

    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(9);
    y = addWrappedText(pdf, message.content || "", 16, y, 178, 5);

    if (Array.isArray(message.citations) && message.citations.length) {
      y += 2;
      pdf.setFont("helvetica", "bold");
      pdf.text("Sources / Citations", 16, y);
      y += 5;
      pdf.setFont("helvetica", "normal");
      message.citations.forEach((citation, citationIndex) => {
        const line =
          typeof citation === "string"
            ? citation
            : citation.label ||
              citation.filename ||
              citation.quoted_text ||
              `Source ${citationIndex + 1}`;
        y = addWrappedText(pdf, `${citationIndex + 1}. ${line}`, 18, y, 174, 4.8);
      });
    }

    if (index < messages.length - 1) {
      y += 3;
      pdf.setDrawColor(80, 90, 110);
      pdf.line(16, y, 194, y);
      y += 6;
    }
  });

  pdf.save(filename);
}
