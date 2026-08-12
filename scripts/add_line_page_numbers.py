"""Add page setup, continuous line numbers, and a page-number footer to a docx.

Pandoc emits a bare body section (no page size, no footer). This post-process
adds, on the body <w:sectPr>: a footer reference + Letter page size + 1in margins
+ continuous line numbers, and creates the footer part (centered PAGE field) plus
its content-type override and relationship. Idempotent.
Usage: python3 add_line_page_numbers.py <path-to.docx>
"""
import re, shutil, sys, zipfile

FTR_RID = "rId990"
FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
    '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'
)
EXTRAS = (
    '<w:pgSz w:w="12240" w:h="15840"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
    'w:header="720" w:footer="720" w:gutter="0"/>'
    '<w:lnNumType w:countBy="1" w:restart="continuous"/>'
    '<w:cols w:space="720"/>'
)

def main(path):
    zin = zipfile.ZipFile(path)
    parts = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    parts["word/footer1.xml"] = FOOTER_XML.encode()

    ct = parts["[Content_Types].xml"].decode()
    if "word/footer1.xml" not in ct:
        ct = ct.replace("</Types>",
            '<Override PartName="/word/footer1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    parts["[Content_Types].xml"] = ct.encode()

    rels = parts["word/_rels/document.xml.rels"].decode()
    if "Target=\"footer1.xml\"" not in rels:
        rels = rels.replace("</Relationships>",
            f'<Relationship Id="{FTR_RID}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
            'Target="footer1.xml"/></Relationships>')
    parts["word/_rels/document.xml.rels"] = rels.encode()

    doc = parts["word/document.xml"].decode()
    ms = list(re.finditer(r"<w:sectPr\b.*?</w:sectPr>", doc, re.S))
    body = ms[-1].group(0)
    s = body
    if "footerReference" not in s:
        s = s.replace("<w:sectPr>", f'<w:sectPr><w:footerReference w:type="default" r:id="{FTR_RID}"/>', 1)
    if "pgSz" not in s:
        if "</w:footnotePr>" in s:
            s = s.replace("</w:footnotePr>", "</w:footnotePr>" + EXTRAS, 1)
        else:
            s = s.replace("</w:sectPr>", EXTRAS + "</w:sectPr>", 1)
    doc = doc[:ms[-1].start()] + s + doc[ms[-1].end():]
    parts["word/document.xml"] = doc.encode()

    tmp = path + ".tmp"
    zo = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
    for n, d in parts.items():
        zo.writestr(n, d if isinstance(d, bytes) else d.encode())
    zo.close()
    shutil.move(tmp, path)
    print("added page setup + continuous line numbers + page-number footer")

if __name__ == "__main__":
    main(sys.argv[1])
