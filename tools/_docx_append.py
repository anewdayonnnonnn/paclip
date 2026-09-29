"""把纯文本追加到《文本测改进方法.docx》末尾（改动存档章节）。

没有 python-docx 可用，走原始 OOXML：读 word/document.xml，在 <w:sectPr 前插入段落 XML，
再整包重写 zip。段落样式照抄文档现有段落（rFonts w:hint="eastAsia"，无 pStyle）。

用法：python _docx_append.py <内容文本文件>
      - 空行 → 空段落
      - 其余每行 → 一个段落（XML 特殊字符自动转义）
每次运行前自动备份为《文本测改进方法_备份_YYYYMMDD.docx》，备份已存在则跳过。
"""
import datetime
import os
import shutil
import sys
import zipfile

DOCX = "文本测改进方法.docx"
HINT = '<w:rFonts w:hint="eastAsia"/>'
PARA = ('<w:p><w:pPr><w:rPr>' + HINT + '</w:rPr></w:pPr>'
        '<w:r><w:rPr>' + HINT + '</w:rPr><w:t xml:space="preserve">{}</w:t></w:r></w:p>')
EMPTY = '<w:p><w:pPr><w:rPr>' + HINT + '</w:rPr></w:pPr></w:p>'


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def backup() -> str:
    stamp = datetime.date.today().strftime("%Y%m%d")
    dst = DOCX.replace(".docx", f"_备份_{stamp}.docx")
    if os.path.exists(dst):
        print(f"[备份] 已存在，跳过 -> {dst}")
    else:
        shutil.copy2(DOCX, dst)
        print(f"[备份] -> {dst}")
    return dst


def append(lines):
    with zipfile.ZipFile(DOCX) as z:
        infos = z.infolist()
        data = {i.filename: z.read(i.filename) for i in infos}

    xml = data["word/document.xml"].decode("utf-8")
    if "<w:sectPr" not in xml:
        raise SystemExit("找不到 <w:sectPr，中止（避免写坏文档）")

    add = "".join(EMPTY if not ln.strip() else PARA.format(esc(ln)) for ln in lines)
    idx = xml.rindex("<w:sectPr")
    xml = xml[:idx] + add + xml[idx:]
    data["word/document.xml"] = xml.encode("utf-8")

    # 保持条目顺序与压缩属性不变
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for i in infos:
            z.writestr(i, data[i.filename])
    print(f"[追加] {len([l for l in lines if l.strip()])} 个段落，document.xml "
          f"{len(xml)} 字符")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    src = sys.argv[1]
    with open(src, encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    backup()
    append(lines)
    # 同步刷新文本快照，便于下次检索
    with zipfile.ZipFile(DOCX) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    import re
    paras = re.findall(r"<w:p\b.*?</w:p>|<w:p\b[^>]*/>", xml, re.S)
    with open("_docx_all.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(paras):
            txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
            f.write(f"{i} | {txt}\n")
    print(f"[快照] _docx_all.txt 共 {len(paras)} 段")


if __name__ == "__main__":
    main()
