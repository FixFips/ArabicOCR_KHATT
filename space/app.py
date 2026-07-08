"""Hugging Face Space demo for ArabicOCR-KHATT.

Deploy: copy this directory's files (app.py, requirements.txt, README.md)
to a Gradio Space. Weights are downloaded from the Hub on startup.
"""

import gradio as gr

from arabicocr_khatt import ArabicOCR

ocr = ArabicOCR.from_pretrained()


def recognize(image, segment, polarity, decoding, upscale):
    if image is None:
        return ""
    beam_width = 10 if decoding.startswith("Beam") else 1
    lm_weight = 0.3 if decoding == "Beam + Arabic bigram LM" else 0.0
    return ocr.recognize(
        image,
        segment=segment,
        beam_width=beam_width,
        lm_weight=lm_weight,
        polarity=polarity,
        upscale=upscale,
    )


with gr.Blocks(title="Arabic Handwritten OCR (KHATT)") as demo:
    gr.Markdown(
        "# ✍️ Arabic Handwritten OCR (KHATT)\n"
        "Line-level Arabic handwriting recognition — CRNN-CTC with Arabic-specific "
        "multi-scale vertical encoding, trained on the KHATT dataset. "
        "[Code on GitHub](https://github.com/FixFips/ArabicOCR_KHATT) · "
        "`pip install arabicocr-khatt`"
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Handwritten Arabic image", type="pil", image_mode="RGB")
            segment = gr.Checkbox(value=True, label="Auto-segment into lines")
            polarity = gr.Radio(
                ["auto", "normal", "invert"], value="auto", label="Polarity",
                info="auto tries both and keeps the reading with more text",
            )
            decoding = gr.Radio(
                ["Greedy", "Beam search", "Beam + Arabic bigram LM"],
                value="Beam + Arabic bigram LM", label="Decoding",
            )
            upscale = gr.Slider(1.0, 3.0, value=1.0, step=0.5, label="Upscale (for tiny text)")
            btn = gr.Button("Recognize", variant="primary")
        with gr.Column():
            output = gr.Textbox(
                label="Recognized text", lines=8, text_align="right", rtl=True,
                show_copy_button=True,
            )

    btn.click(recognize, [image, segment, polarity, decoding, upscale], output)


if __name__ == "__main__":
    demo.launch()
