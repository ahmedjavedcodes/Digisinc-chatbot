filename = "Agency_Profile.md"

with open(filename, 'r', encoding='utf-8') as file:
    content = file.read()
    char_count = len(content)

print(f"Total characters (including Markdown syntax): {char_count}")