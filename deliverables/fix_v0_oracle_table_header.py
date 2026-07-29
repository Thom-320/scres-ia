#!/usr/bin/env python3
"""Repeat the Section 3.4 results-table header across Word page breaks."""

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


DOCX = (
    Path(__file__).resolve().parent
    / "v0_neuralNet-scres_DES_and_oracle_metric.docx"
)


def main() -> None:
    document = Document(DOCX)
    matches = [
        table
        for table in document.tables
        if table.rows
        and "Controller" in " ".join(cell.text for cell in table.rows[0].cells)
        and any(
            "Clairvoyant ceiling" in cell.text
            for row in table.rows
            for cell in row.cells
        )
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one oracle table, found {len(matches)}")
    table = matches[0]
    for index, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        if tr_pr.find(qn("w:cantSplit")) is None:
            tr_pr.append(OxmlElement("w:cantSplit"))
        if index == 0 and tr_pr.find(qn("w:tblHeader")) is None:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
    document.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    main()
