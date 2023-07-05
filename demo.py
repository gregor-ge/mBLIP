# based on demo from https://huggingface.co/spaces/hysts/BLIP2-with-transformers/blob/main/app.py
import contextlib
import os
import string
import argparse
import gradio as gr
import PIL.Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

parser = argparse.ArgumentParser()

parser.add_argument("--cpu", action="store_true", default=False)

args = parser.parse_args()

DESCRIPTION = '# [mBLIP](https://github.com/gregor-ge/mBLIP)'

if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

MODEL_ID = "Gregor/mblip-mt0-xl"

if torch.cuda.is_available() and not args.cpu:
    model_info = {
            'processor':
            AutoProcessor.from_pretrained(MODEL_ID),
            'model':
            Blip2ForConditionalGeneration.from_pretrained(MODEL_ID,
                                                          device_map='auto',
                                                          load_in_8bit=True),
        }
else:
    model_info = {
            'processor':
            AutoProcessor.from_pretrained(MODEL_ID),
            'model':
            Blip2ForConditionalGeneration.from_pretrained(MODEL_ID),
        }

def answer_question(image: PIL.Image.Image, text: str,
                    decoding_method: str, temperature: float, num_beams: int,
                    length_penalty: float, repetition_penalty: float) -> str:
    processor = model_info['processor']
    model = model_info['model']
    length_penalty = float(length_penalty)
    inputs = processor(images=image, text=text,
                       return_tensors='pt')
    context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if torch.cuda.is_available() and not args.cpu else contextlib.nullcontext()
    with context:
        generated_ids = model.generate(**inputs,
                                       do_sample=decoding_method ==
                                       'Nucleus sampling',
                                       temperature=temperature,
                                       length_penalty=length_penalty,
                                       repetition_penalty=repetition_penalty,
                                       max_length=256,
                                       min_length=1,
                                       num_beams=num_beams,
                                       top_p=0.9)
    result = processor.batch_decode(generated_ids,
                                    skip_special_tokens=True)[0].strip()
    return result


def postprocess_output(output: str) -> str:
    # if output and not output[-1] in string.punctuation:
    #     output += '.'
    return output


def chat(
    image: PIL.Image.Image,
    text: str,
    decoding_method: str,
    use_history: str,
    temperature: float,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    history: list[str] = [],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:

    if use_history == "No":
        history = []
    history.append(text)
    prompt = '\n'.join(history)

    output = answer_question(
        image,
        prompt,
        decoding_method,
        temperature,
        num_beams,
        length_penalty,
        repetition_penalty,
    )
    output = postprocess_output(output)
    history.append(output)

    chat_val = list(zip(history[0::2], history[1::2]))
    return gr.update(value=chat_val), gr.update(value=history)


examples = [
    [
        'Krk_waterfalls.jpg',
        'Describe the image in Croatian.',
    ],
    [
        'Krk_waterfalls.jpg',
        'Opišite sliku što detaljnije.',
    ],
]

with gr.Blocks() as demo:   #css='style.css'
    gr.Markdown(DESCRIPTION)

    image = gr.Image(type='pil')
    with gr.Accordion(label='Advanced settings', open=False):
        sampling_method = gr.Radio(
            label='Text Decoding Method',
            choices=['Beam search', 'Nucleus sampling'],
            value='Beam search',
        )
        use_history = gr.Radio(
            label='Include previous input/output to instruction',
            choices=['No', 'Yes (untested)'],
            value='No',
        )
        temperature = gr.Slider(
            label='Temperature (used with nucleus sampling)',
            minimum=0.5,
            maximum=1.0,
            value=1.0,
            step=0.1,
        )
        num_beams = gr.Slider(
            label='Number of beams',
            minimum=1,
            maximum=5,
            value=5,
            step=1,
        )
        length_penalty = gr.Slider(
            label=
            'Length Penalty (set to larger for longer sequence, used with beam search)',
            minimum=-1.0,
            maximum=2.0,
            value=1.0,
            step=0.2,
        )
        rep_penalty = gr.Slider(
            label='Repeat Penalty (larger value prevents repetition)',
            minimum=1.0,
            maximum=5.0,
            value=1.5,
            step=0.5,
        )
    with gr.Row():
        with gr.Box():
            chatbot = gr.Chatbot(label='Prompt the model')
            history = gr.State(value=[])
            instruct_input = gr.Text(label='Instruction',
                                show_label=False,
                                max_lines=1).style(container=False)
            with gr.Row():
                clear_chat_button = gr.Button(value='Clear')
                chat_button = gr.Button(value='Submit')

    gr.Examples(
        examples=examples,
        inputs=[
            image,
            instruct_input,
        ],
    )

    chat_inputs = [
        image,
        instruct_input,
        sampling_method,
        use_history,
        temperature,
        num_beams,
        length_penalty,
        rep_penalty,
        history,
    ]
    chat_outputs = [
        chatbot,
        history,
    ]
    instruct_input.submit(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )
    chat_button.click(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
        api_name='chat',
    )
    clear_chat_button.click(
        fn=lambda: ('', [], [], []),
        inputs=None,
        outputs=[
            instruct_input,
            chatbot,
            history,
        ],
        queue=False,
        api_name='clear',
    )
    image.change(
        fn=lambda: ('', [], []),
        inputs=None,
        outputs=[
            chatbot,
            history,
        ],
        queue=False,
    )

demo.queue(max_size=10).launch(share=True)

