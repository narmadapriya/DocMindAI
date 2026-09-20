export const formatBytes = (value) => {
    const bytes = Number(value || 0);
    if (!bytes)
        return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
};
export const formatDate = (value) => {
    if (!value)
        return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime())
        ? "—"
        : date.toLocaleDateString(undefined, { month: "short", day: "2-digit", year: "numeric" });
};
export const initials = (name) => (name || "User")
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

function citationField(citation, label) {
    const pattern = new RegExp(
        `\\b${label}:?\\s*(.*?)(?=,\\s*(?:Page|Section|Sheet|Table):?|\\])`,
        "i",
    );
    return citation.match(pattern)?.[1]?.trim() || null;
}

export function normalizeCitation(citation, index = 0) {
    if (typeof citation !== "string")
        return citation;

    const page = citation.match(/\bPage\s+(\d+)/i)?.[1];
    const filename = citation.match(/\[Source:\s*([^,\]]+)/i)?.[1]?.trim();
    const section = citationField(citation, "Section");
    const sheet = citationField(citation, "Sheet");
    const table = citationField(citation, "Table");

    return {
        id: `citation-${index}`,
        filename,
        page_number: page ? Number(page) : null,
        section_name: section,
        sheet_name: sheet,
        table_name: table,
        label: citation,
    };
}
export function documentReady(doc) {
    if (doc.ready_for_rag === true)
        return true;
    return ["ready", "processed", "success", "indexed"].includes((doc.status || "").toLowerCase());
}
