/**
 * Render reports/REPORT.md to a polished Word document.
 *
 *   node src/make_docx.js
 *
 * Handles the constructs the report actually uses: ATX headings, GFM pipe
 * tables, images, fenced code, blockquotes, bullet/ordered lists, horizontal
 * rules, and inline bold / italic / code / links.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, ShadingType,
  LevelFormat, convertInchesToTwip,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const REPORTS = path.join(ROOT, "reports");
const MD = path.join(REPORTS, "REPORT.md");
const OUT = path.join(REPORTS, "Week1_Preprocessing_Report.docx");

// A4 (11906 DXA) minus 0.75" margins each side.
const MARGIN = convertInchesToTwip(0.75);
const CONTENT_DXA = 11906 - 2 * MARGIN;
const MAX_IMG_PX = 640;

const INK = "15140F", INK2 = "3D3B35", MUTED = "6F6C64";
const ACCENT = "2A78D6", RULE = "D9D7D0", SOFT = "F5F4F1";
const SERIF = "Cambria", SANS = "Calibri", MONO = "Consolas";

/* ------------------------------------------------------------------ inline */
const CODE_MARK = "\u0001";

function inlineRuns(text, base = {}) {
  const runs = [];

  // Phase 1 (top-level only): mask code spans, so an asterisk inside one -
  // e.g. `outputs/05_*.csv` - can never be mistaken for an emphasis marker.
  let codes = base.codes;
  if (!codes) {
    codes = [];
    text = text.replace(/`([^`]+)`/g, (_, c) => {
      codes.push(c);
      return `${CODE_MARK}${codes.length - 1}${CODE_MARK}`;
    });
    base = { ...base, codes };
  }

  const re = /(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)]+\))|(\*[^*\n]+\*)/g;
  let last = 0, m;

  const codeOpts = () => (base.inheritCode
    ? { font: MONO }
    : { font: MONO, size: 18, color: "9A3412" });

  const emit = (t, opts) => {
    if (!t) return;
    runs.push(new TextRun({
      text: t, font: base.font || SERIF, size: base.size || 21,
      color: base.color || INK, bold: base.bold, italics: base.italics, ...opts,
    }));
  };

  // Phase 2: expand masked code spans back out while emitting literal text.
  const push = (t, opts) => {
    if (!t) return;
    const parts = t.split(new RegExp(`${CODE_MARK}(\\d+)${CODE_MARK}`));
    parts.forEach((part, idx) => {
      if (idx % 2 === 1) emit(codes[Number(part)], { ...opts, ...codeOpts() });
      else emit(part, opts);
    });
  };
  const unescape = (s) => s.replace(/\\\|/g, "|").replace(/\\\*/g, "*");

  while ((m = re.exec(text)) !== null) {
    push(unescape(text.slice(last, m.index)));
    if (m[1]) {
      // Recurse so a code span nested inside **bold** is still formatted.
      runs.push(...inlineRuns(m[1].slice(2, -2), { ...base, bold: true }));
    } else if (m[2]) {
      const label = m[2].slice(1, m[2].indexOf("]"));
      push(unescape(label), { color: ACCENT, underline: {} });
    } else if (m[3]) {
      runs.push(...inlineRuns(m[3].slice(1, -1), { ...base, italics: true }));
    }
    last = re.lastIndex;
  }
  push(unescape(text.slice(last)));
  return runs.length ? runs : [new TextRun({ text: "", font: SERIF, size: 21 })];
}

/* ------------------------------------------------------------------- images */
function pngSize(buf) {
  // IHDR width/height are big-endian uint32 at bytes 16 and 20.
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function imageParagraph(src) {
  const file = path.join(REPORTS, src);
  if (!fs.existsSync(file)) {
    console.warn("  WARNING: missing image " + src);
    return null;
  }
  const buf = fs.readFileSync(file);
  const { w, h } = pngSize(buf);
  const width = Math.min(MAX_IMG_PX, w);
  const height = Math.round(h * (width / w));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 240 },
    keepLines: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width, height } })],
  });
}

/* ------------------------------------------------------------------- tables */
function splitRow(line) {
  const cells = [];
  let cur = "";
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "\\" && line[i + 1] === "|") { cur += "\\|"; i++; continue; }
    if (line[i] === "|") { cells.push(cur); cur = ""; continue; }
    cur += line[i];
  }
  cells.push(cur);
  return cells.slice(1, -1).map((c) => c.trim());
}

function buildTable(rows) {
  const header = splitRow(rows[0]);
  const body = rows.slice(2).map(splitRow);
  const n = header.length;

  // Width each column by its widest content, clamped so nothing collapses.
  const weights = header.map((h, i) => {
    const lens = [h.length, ...body.map((r) => (r[i] || "").length)];
    return Math.max(6, Math.min(60, Math.max(...lens)));
  });
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / total) * CONTENT_DXA));
  widths[n - 1] += CONTENT_DXA - widths.reduce((a, b) => a + b, 0); // exact sum

  const border = { style: BorderStyle.SINGLE, size: 2, color: RULE };
  const cell = (text, i, isHeader) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: SOFT, color: "auto" } : undefined,
    borders: { top: border, bottom: border, left: border, right: border },
    children: [new Paragraph({
      spacing: { before: 0, after: 0 },
      children: inlineRuns(text, {
        font: SANS, size: 17, color: isHeader ? INK : INK2, bold: isHeader || undefined,
      }),
    })],
  });

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((h, i) => cell(h, i, true)),
      }),
      ...body.map((r) => new TableRow({
        children: Array.from({ length: n }, (_, i) => cell(r[i] || "", i, false)),
      })),
    ],
  });
}

/* --------------------------------------------------------------- the parser */
function parse(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;

  const heading = (text, level) => {
    const sizes = { 1: 40, 2: 27, 3: 22 };
    return new Paragraph({
      heading: { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3 }[level],
      spacing: { before: level === 1 ? 0 : level === 2 ? 400 : 260, after: level === 1 ? 80 : 130 },
      keepNext: true,
      border: level === 2
        ? { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 6 } }
        : undefined,
      // Parse inline markup so `code` spans in headings don't show backticks.
      children: inlineRuns(text, {
        font: SANS, size: sizes[level], bold: true, inheritCode: true,
        color: level === 3 ? INK2 : INK,
      }),
    });
  };

  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (!t) { i++; continue; }

    // horizontal rule
    if (/^---+$/.test(t)) {
      out.push(new Paragraph({
        spacing: { before: 120, after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 1 } },
        children: [new TextRun({ text: "", size: 2 })],
      }));
      i++; continue;
    }

    // headings
    const h = t.match(/^(#{1,3})\s+(.*)$/);
    if (h) { out.push(heading(h[2], h[1].length)); i++; continue; }

    // fenced code
    if (t.startsWith("```")) {
      i++;
      const buf = [];
      while (i < lines.length && !lines[i].trim().startsWith("```")) buf.push(lines[i++]);
      i++;
      buf.forEach((cl, k) => out.push(new Paragraph({
        shading: { type: ShadingType.CLEAR, fill: SOFT, color: "auto" },
        spacing: { before: k === 0 ? 80 : 0, after: k === buf.length - 1 ? 200 : 0 },
        indent: { left: 170, right: 170 },
        keepLines: true,
        border: {
          left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 6 },
          ...(k === 0 ? { top: { style: BorderStyle.SINGLE, size: 2, color: SOFT, space: 2 } } : {}),
          ...(k === buf.length - 1 ? { bottom: { style: BorderStyle.SINGLE, size: 2, color: SOFT, space: 2 } } : {}),
        },
        children: [new TextRun({ text: cl || " ", font: MONO, size: 16, color: INK })],
      })));
      continue;
    }

    // tables
    if (t.startsWith("|") && i + 1 < lines.length && /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++].trim());
      out.push(buildTable(rows));
      out.push(new Paragraph({ spacing: { after: 220 }, children: [new TextRun({ text: "", size: 2 })] }));
      continue;
    }

    // standalone image
    const img = t.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
    if (img) { const p = imageParagraph(img[1]); if (p) out.push(p); i++; continue; }

    // blockquote
    if (t.startsWith(">")) {
      const buf = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        buf.push(lines[i].trim().replace(/^>\s?/, "")); i++;
      }
      out.push(new Paragraph({
        shading: { type: ShadingType.CLEAR, fill: SOFT, color: "auto" },
        spacing: { before: 120, after: 220 },
        indent: { left: 170, right: 170 },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 6 } },
        children: inlineRuns(buf.join(" "), { color: INK2 }),
      }));
      continue;
    }

    // lists
    const li = t.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (li) {
      const ordered = /^\d+\./.test(t);
      const buf = [];
      while (i < lines.length) {
        const m2 = lines[i].trim().match(/^(?:[-*]|\d+\.)\s+(.*)$/);
        if (!m2) {
          // continuation line of the current bullet
          if (lines[i].startsWith("  ") && lines[i].trim() && buf.length) {
            buf[buf.length - 1] += " " + lines[i].trim(); i++; continue;
          }
          break;
        }
        buf.push(m2[1]); i++;
      }
      buf.forEach((item) => out.push(new Paragraph({
        numbering: { reference: ordered ? "ol" : "ul", level: 0 },
        spacing: { before: 40, after: 60 },
        children: inlineRuns(item),
      })));
      out.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "", size: 2 })] }));
      continue;
    }

    // paragraph (soft-wrapped across lines)
    const buf = [];
    while (i < lines.length) {
      const s = lines[i].trim();
      if (!s || /^(#{1,3}\s|---+$|\||>|```|!\[)/.test(s) || /^(?:[-*]|\d+\.)\s/.test(s)) break;
      buf.push(s); i++;
    }
    out.push(new Paragraph({
      spacing: { after: 160, line: 300 },
      alignment: AlignmentType.LEFT,
      children: inlineRuns(buf.join(" ")),
    }));
  }
  return out;
}

/* ----------------------------------------------------------------- assemble */
const md = fs.readFileSync(MD, "utf8");
const children = parse(md);

const doc = new Document({
  creator: "Nischay Prasad",
  title: "Week 1 - Python for Machine Learning & Data Preprocessing",
  description: "Cleaning and preprocessing the UCI Adult census dataset.",
  numbering: {
    config: [
      {
        reference: "ul",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 220 } } },
        }],
      },
      {
        reference: "ol",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 220 } } },
        }],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: SERIF, size: 21, color: INK } },
    },
  },
  sections: [{
    properties: { page: { margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`DOCX -> ${OUT}  (${(buf.length / 1024).toFixed(0)} KB, ${children.length} blocks)`);
});
