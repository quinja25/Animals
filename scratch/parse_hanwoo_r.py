import re

with open("C:/Users/jaeyo/.gemini/antigravity-cli/brain/05186fea-be0e-45e3-935d-75dccfde68a3/.system_generated/steps/200/content.md", "r", encoding="utf-8") as f:
    c = f.read()

links = set(re.findall(r'href="/adatalab/hanwoo/blob/[^"]+"', c))
print("Files in R/ directory:")
for l in links:
    print(l)
