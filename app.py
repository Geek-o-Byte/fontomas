import gradio as gr

font_generated = gr.State(False)


def show_message(message):
    return gr.Markdown(message)


def generate_font(reference=None):
    # Placeholder for generated font string
    return "Generated Font String"


# Gradio Blocks app
with gr.Blocks() as iface:
    tru = False
    # Header
    gr.Markdown("<h1>Font Generator</h1>")

    # Description
    gr.Markdown("Generate a new font based on a reference or randomly.")

    name = gr.Textbox(label="Font name", value="Font-Proto", interactive=True)

    generate_button = gr.Button("Generate")

    # Use vertical stacking by default within Blocks
    # (Optional visual customization can be done with CSS)

    # Events
    generate_button.click()

    gr.Markdown("Test a font")
    area = gr.HTML("""
    <style>
    @font-face {
        font-family: "widerandom";
        src: url("/file=widerandom.ttf") format("truetype");
    }

    .feedback {font-size: 90px !important; 
    color: black;
    height: 20vh !important;
    font-family: "widerandom" !important;
    padding: 2rem !important;
    letter-spacing: 1rem !important;
    text-transform: uppercase !important;
    }
    </style>
    <textarea class="feedback">
  
    </textarea>""")

    font_ufo = None
    font_ttf = None
    font_otf = None
    font_woff = None

    with gr.Row():
        down_ttf = gr.DownloadButton("Download TTF", value=font_ttf)
        down_otf = gr.DownloadButton("Download OTF", value=font_otf)
        down_woff = gr.DownloadButton("Download WOFF", value=font_woff)
        down_ufo = gr.DownloadButton("Download UFO", value=font_ufo)

with gr.Blocks() as iface:
    gr.Markdown("<h1>Font Generator with reference</h1>")

    # Description
    gr.Markdown("Generate a new font based on a reference or randomly.")

    style_block = gr.HTML("""
      <style>
        textarea{
          font-size: 90px;
        }
      </style>
      """)

    gr.Textbox(label="Font name", value="Font-Proto", interactive=True)

    generate_button = gr.Button("Generate")

    generate_button.click(lambda: show_message(generate_font(reference=None)))

    textarea = gr.TextArea("", label="Test a font",
                           interactive=True)

with gr.Blocks() as iface_ref:
    gr.Markdown("<h1>Font Generator with reference</h1>")

    # Description
    gr.Markdown("Generate a new font based on a reference or randomly.")

    style_block = gr.HTML("""
      <style>
        textarea{
          font-size: 90px;
        }
      </style>
      """)

    gr.Textbox(label="Font name", value="Font-Proto", interactive=True)

    file_upload = gr.File(label="Upload Reference Font (optional)")

    # Generate Button

    generate_button = gr.Button("Generate")

    generate_button.click(lambda: show_message(generate_font(reference=file_upload.value)))

    textarea = gr.TextArea("", label="Test a font",
                           interactive=True)

    with gr.Row():
        down_ttf = gr.DownloadButton("Download TTF")
        down_otf = gr.DownloadButton("Download OTF")
        down_woff = gr.DownloadButton("Download WOFF")
        down_ufo = gr.DownloadButton("Download UFO")

with gr.Blocks() as about:
    gr.Markdown("<h1>About app</h1>")

    gr.Image("preview.jpg", width=400, show_label=False, show_download_button=False)
    # Description
    gr.Markdown("Generate a new font based on a reference or randomly.")

with gr.Blocks(theme='bethecloud/storj_theme') as main:
    gr.Markdown("# FONT GENERATOR / DEMO")
    demo = gr.TabbedInterface([iface, iface_ref, about], ["Font Generation", "Reference generation", "About app"])
    gr.Markdown("***Created by Max Maslov | 2024***")

# Launch the app
main.launch(allowed_paths=["widerandom.ttf"])
