filename = "Agency_Profile.md"

with open(filename, 'r', encoding='utf-8') as file:
    content = file.read()
    char_count = len(content)

print(f"Total characters (including Markdown syntax): {char_count}")


output = """
**Digisinc's Services:**

We provide a range of services to help you achieve your digital goals. Our expertise includes:

* **Websites & Apps**: We engineer custom, high-performance web applications that are fast, reliable, and optimized for seamless cross-device experiences.
* **High Conversion Landing Pages**: Our team creates landing pages that drive conversions and help you achieve your business objectives.
* **CMS Integrated Websites**: We design and develop websites that are integrated with Content Management Systems (CMS) for easy content management and updates.
* **Full Stack Web Apps & E-commerce**: Our team builds full-stack web applications and e-commerce solutions that meet your business needs.

Let us know if you'd like to start a project. You can reach out to us at +92 317 8433864 or digisinc.systems@gmail.com.
"""

print(len(output))