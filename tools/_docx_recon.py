"""侦察 docx 结构：为按原始 OOXML 追加段落做准备（无 python-docx，只能手写 XML）。"""
import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

PATH = "文本测改进方法.docx"
with zipfile.ZipFile(PATH) as z:
    names = z.namelist()
    xml = z.read("word/document.xml").decode("utf-8")

print(f"zip 条目 {len(names)} 个：")
for n in names:
    print(f"   {n}")
print(f"\ndocument.xml 长度 {len(xml)} 字符")

body = xml[xml.index("<w:body>"):]
print(f"\n=== body 尾部 1500 字符 ===")
print(body[-1500:])

paras = re.findall(r"<w:p\b.*?</w:p>|<w:p\b[^>]*/>", xml, re.S)
print(f"\n=== 共 {len(paras)} 个 w:p ===")
for i, p in enumerate(paras[-8:], start=len(paras) - 8):
    txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
    print(f"\n--- p[{i}] 文本={txt[:70]!r}")
    print(p[:900])
